"""
go2_unitree_guide.py
====================
A faithful Python/MuJoCo port of the unitree_guide trotting controller
(originally written in C++ for Go1/A1 with Gazebo/ROS).

Ported modules (mirroring the original C++ structure):
  * QuadrupedLeg   – FK, IK, Jacobian      (unitreeLeg.cpp)
  * Go2Robot       – full-robot wrappers    (unitreeRobot.cpp)
  * WaveGenerator  – contact/phase clock   (WaveGenerator.cpp)
  * FeetEndCal     – next foothold calc     (FeetEndCal.cpp)
  * GaitGenerator  – cycloid swing traj     (GaitGenerator.cpp)
  * BalanceCtrl    – QP force solver        (BalanceCtrl.cpp)
  * Trotting FSM   – main run loop          (State_Trotting.cpp)

Go2 parameters extracted directly from go2.xml (unitree_mujoco).

Requirements:
    pip install mujoco scipy numpy

Usage:
    python go2_unitree_guide.py
    (close the viewer window to quit)
"""

import math
import time
import numpy as np
from scipy.optimize import minimize
import mujoco
import mujoco.viewer

# ==============================================================================
# Go2 physical constants  (extracted from go2.xml)
# ==============================================================================
ABAD_LEN   = 0.0955   # hip abduction link length [m]
THIGH_LEN  = 0.213    # upper leg length [m]
CALF_LEN   = 0.213    # lower leg length [m]

# Hip joint origins in body frame: order FR, FL, RR, RL
HIP_POS = np.array([
    [ 0.1934, -0.0955, 0.0],   # FR  (right → sideSign = -1 per unitreeLeg.cpp)
    [ 0.1934,  0.0955, 0.0],   # FL  (left  → sideSign = +1)
    [-0.1934, -0.0955, 0.0],   # RR  (right → sideSign = -1)
    [-0.1934,  0.0955, 0.0],   # RL  (left  → sideSign = +1)
])

# Nominal standing foot positions in body frame  shape (3,4)
FEET_NORMAL_STAND = np.array([
    [ 0.1934,  0.1934, -0.1934, -0.1934],
    [-0.1300,  0.1300, -0.1300,  0.1300],
    [-0.3200, -0.3200, -0.3200, -0.3200],
])

ROBOT_MASS   = 15.0
ROB_INERTIA  = np.diag([0.132, 0.3475, 0.3775])

# Joint actuator names matching go2.xml actuator names
JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

ACTUATOR_NAMES = [n.replace("_joint","") for n in JOINT_NAMES]

# ==============================================================================
# Math helpers
# ==============================================================================

def skew(v):
    return np.array([[ 0,    -v[2],  v[1]],
                     [ v[2],  0,    -v[0]],
                     [-v[1],  v[0],  0   ]])

def rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def quat_to_rotmat_wxyz(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])

def rotmat_to_exp(R):
    theta = math.acos(max(-1.0,min(1.0,(np.trace(R)-1)/2)))
    if abs(theta)<1e-8: return np.zeros(3)
    return theta/(2*math.sin(theta))*np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])

def sat(x,lo,hi): return max(lo,min(hi,x))

# ==============================================================================
# QuadrupedLeg  ←  unitreeLeg.cpp
# ==============================================================================

class QuadrupedLeg:
    def __init__(self, leg_id, abad, thigh, calf, p_hip2b):
        self.la = abad; self.lt = thigh; self.lc = calf
        self.p  = np.array(p_hip2b)
        self.ss = -1 if leg_id in (0,2) else 1   # sideSign

    # FK: joint angles -> foot pos in hip frame
    def calc_pEe2H(self, q):
        l1,l2,l3 = self.ss*self.la, -self.lt, -self.lc
        s1,c1 = math.sin(q[0]),math.cos(q[0])
        s2,c2 = math.sin(q[1]),math.cos(q[1])
        s3,c3 = math.sin(q[2]),math.cos(q[2])
        c23,s23 = c2*c3-s2*s3, s2*c3+c2*s3
        return np.array([
            l3*s23+l2*s2,
            -l3*s1*c23+l1*c1-l2*c2*s1,
             l3*c1*c23+l1*s1+l2*c1*c2])

    def calc_pEe2B(self, q):
        return self.p + self.calc_pEe2H(q)

    # Jacobian
    def calc_jaco(self, q):
        l1,l2,l3 = self.la*self.ss, -self.lt, -self.lc
        s1,c1 = math.sin(q[0]),math.cos(q[0])
        s2,c2 = math.sin(q[1]),math.cos(q[1])
        s3,c3 = math.sin(q[2]),math.cos(q[2])
        c23,s23 = c2*c3-s2*s3, s2*c3+c2*s3
        J = np.zeros((3,3))
        J[0,0]=0;               J[1,0]=-l3*c1*c23-l2*c1*c2-l1*s1; J[2,0]=-l3*s1*c23-l2*c2*s1+l1*c1
        J[0,1]=l3*c23+l2*c2;   J[1,1]= l3*s1*s23+l2*s1*s2;        J[2,1]=-l3*c1*s23-l2*c1*s2
        J[0,2]=l3*c23;          J[1,2]= l3*s1*s23;                  J[2,2]=-l3*c1*s23
        return J

    def calc_vEe(self, q, qd): return self.calc_jaco(q)@qd

    # IK helpers
    def _q1(self,py,pz,l1):
        L=math.sqrt(max(0,py**2+pz**2-l1**2))
        return math.atan2(pz*l1+py*L, py*l1-pz*L)
    def _q3(self,b3,b4,b):
        t=max(-1,min(1,(b3**2+b4**2-b**2)/(2*abs(b3*b4))))
        return -(math.pi-math.acos(t))
    def _q2(self,q1,q3,px,py,pz,b3,b4):
        a1=py*math.sin(q1)-pz*math.cos(q1); a2=px
        m1=b4*math.sin(q3); m2=b3+b4*math.cos(q3)
        return math.atan2(m1*a1+m2*a2, m1*a2-m2*a1)

    def calc_q(self, pEe, body_frame=False):
        p=np.array(pEe)
        if body_frame: p=p-self.p
        px,py,pz=p
        b2y=self.la*self.ss; b3=-self.lt; b4=-self.lc
        c=math.sqrt(px**2+py**2+pz**2)
        b=math.sqrt(max(0,c**2-self.la**2))
        q1=self._q1(py,pz,b2y)
        q3=self._q3(b3,b4,b)
        q2=self._q2(q1,q3,px,py,pz,b3,b4)
        return np.array([q1,q2,q3])

    def calc_qd(self, q, vEe):
        return np.linalg.solve(self.calc_jaco(q), vEe)

    def calc_tau(self, q, f):
        return self.calc_jaco(q).T @ f

# ==============================================================================
# Go2Robot  ←  unitreeRobot.cpp
# ==============================================================================

class Go2Robot:
    def __init__(self):
        self.legs=[QuadrupedLeg(i,ABAD_LEN,THIGH_LEN,CALF_LEN,HIP_POS[i]) for i in range(4)]
        self.feet_normal_stand=FEET_NORMAL_STAND.copy()
        self.mass=ROBOT_MASS; self.inertia=ROB_INERTIA.copy(); self.pcb=np.zeros(3)

    def get_q(self, fp_body):      # fp_body (3,4)
        q=np.zeros(12)
        for i in range(4): q[3*i:3*i+3]=self.legs[i].calc_q(fp_body[:,i],True)
        return q

    def get_qd(self, fp_body, fv_body):
        qd=np.zeros(12)
        for i in range(4):
            qi=self.legs[i].calc_q(fp_body[:,i],True)
            qd[3*i:3*i+3]=self.legs[i].calc_qd(qi,fv_body[:,i])
        return qd

    def get_tau(self, q12, f_body):
        tau=np.zeros(12)
        for i in range(4): tau[3*i:3*i+3]=self.legs[i].calc_tau(q12[3*i:3*i+3],f_body[:,i])
        return tau

    def get_feet_pos_body(self, q12):
        fp=np.zeros((3,4))
        for i in range(4): fp[:,i]=self.legs[i].calc_pEe2B(q12[3*i:3*i+3])
        return fp

    def get_feet_vel_body(self, q12, qd12):
        fv=np.zeros((3,4))
        for i in range(4): fv[:,i]=self.legs[i].calc_vEe(q12[3*i:3*i+3],qd12[3*i:3*i+3])
        return fv

# ==============================================================================
# WaveGenerator  ←  WaveGenerator.cpp
# ==============================================================================

class WaveGenerator:
    def __init__(self, period=0.5, stance_ratio=0.5, bias=None):
        self.T=period; self.sr=stance_ratio
        self.bias=bias if bias is not None else np.array([0.0,0.5,0.5,0.0])
        self._t0=time.perf_counter()
    def get_t_stance(self): return self.T*self.sr
    def get_t_swing(self):  return self.T*(1-self.sr)
    def calc(self):
        t=time.perf_counter()-self._t0
        ph=np.zeros(4); ct=np.zeros(4,int)
        for i in range(4):
            nt=math.fmod(t+self.T-self.T*self.bias[i],self.T)/self.T
            if nt<self.sr: ct[i]=1; ph[i]=nt/self.sr
            else:          ct[i]=0; ph[i]=(nt-self.sr)/(1-self.sr)
        return ph,ct
    def reset(self): self._t0=time.perf_counter()

# ==============================================================================
# FeetEndCal  ←  FeetEndCal.cpp
# ==============================================================================

class FeetEndCal:
    def __init__(self, t_stance, t_swing):
        self.ts=t_stance; self.tw=t_swing
        self.kx=self.ky=self.kyaw=0.005
        fp=FEET_NORMAL_STAND
        self.radii=np.sqrt(fp[0,:]**2+fp[1,:]**2)
        self.angles=np.arctan2(fp[1,:],fp[0,:])

    def cal(self, i, body_pos, body_vel, yaw, dyaw, vxy_goal, dyaw_goal, phase):
        ns=np.zeros(3)
        ns[0]=(body_vel[0]*(1-phase)*self.tw+body_vel[0]*self.ts/2
               +self.kx*(body_vel[0]-vxy_goal[0]))
        ns[1]=(body_vel[1]*(1-phase)*self.tw+body_vel[1]*self.ts/2
               +self.ky*(body_vel[1]-vxy_goal[1]))
        ny=(dyaw*(1-phase)*self.tw+dyaw*self.ts/2+self.kyaw*(dyaw_goal-dyaw))
        ns[0]+=self.radii[i]*math.cos(yaw+self.angles[i]+ny)
        ns[1]+=self.radii[i]*math.sin(yaw+self.angles[i]+ny)
        fp=body_pos+ns; fp[2]=0.0
        return fp

# ==============================================================================
# GaitGenerator  ←  GaitGenerator.cpp
# ==============================================================================

class GaitGenerator:
    def __init__(self, wave, fc):
        self.wave=wave; self.fc=fc
        self.tw=wave.get_t_swing()
        self.h=0.08; self.vxy=np.zeros(2); self.dYaw=0.0
        self._first=True
        self._sP=np.zeros((3,4)); self._eP=np.zeros((3,4))
        self._past_ph=np.zeros(4)

    def set_gait(self, vxy, dYaw, h):
        self.vxy=vxy; self.dYaw=dYaw; self.h=h

    def restart(self):
        self._first=True; self.vxy[:]=0.0

    def run(self, feet_global, body_pos, body_vel, yaw, dyaw, phase, contact):
        if self._first:
            self._sP=feet_global.copy(); self._first=False
        fpos=np.zeros((3,4)); fvel=np.zeros((3,4))
        for i in range(4):
            if contact[i]:
                if self._past_ph[i]<0.5: self._sP[:,i]=feet_global[:,i]
                fpos[:,i]=self._sP[:,i]
            else:
                self._eP[:,i]=self.fc.cal(i,body_pos,body_vel,yaw,dyaw,
                                           self.vxy,self.dYaw,phase[i])
                fpos[:,i]=self._spos(i,phase[i])
                fvel[:,i]=self._svel(i,phase[i])
        self._past_ph=phase.copy()
        return fpos,fvel

    def _cyc_pos(self,s,e,ph):
        p=2*math.pi*ph; return (e-s)*(p-math.sin(p))/(2*math.pi)+s
    def _cyc_vel(self,s,e,ph):
        p=2*math.pi*ph; return (e-s)*(1-math.cos(p))/self.tw
    def _cyc_z_pos(self,s,h,ph):
        p=2*math.pi*ph; return h*(1-math.cos(p))/2+s
    def _cyc_z_vel(self,h,ph):
        p=2*math.pi*ph; return h*math.pi*math.sin(p)/self.tw

    def _spos(self,i,ph):
        s,e=self._sP[:,i],self._eP[:,i]
        return np.array([self._cyc_pos(s[0],e[0],ph),
                         self._cyc_pos(s[1],e[1],ph),
                         self._cyc_z_pos(s[2],self.h,ph)])
    def _svel(self,i,ph):
        s,e=self._sP[:,i],self._eP[:,i]
        return np.array([self._cyc_vel(s[0],e[0],ph),
                         self._cyc_vel(s[1],e[1],ph),
                         self._cyc_z_vel(self.h,ph)])

# ==============================================================================
# BalanceCtrl  ←  BalanceCtrl.cpp  (scipy SLSQP instead of quadprog++)
# ==============================================================================

class BalanceCtrl:
    def __init__(self, rob):
        self.mass=rob.mass; self.Ib=rob.inertia; self.pcb=rob.pcb
        self.g=np.array([0,0,-9.81]); self.mu=0.4
        s=np.array([20,20,50,450,450,450])
        self.S=np.diag(s)
        self.W=np.diag(np.tile([10,10,4],4))
        self.U=np.diag(np.full(12,3.0))
        self.alpha=0.001; self.beta=0.1
        self._Fp=np.zeros(12)

    def cal_f(self, ddPcd, dWbd, R, fp_body, contact):
        A=np.zeros((6,12))
        for i in range(4):
            A[0:3,3*i:3*i+3]=np.eye(3)
            A[3:6,3*i:3*i+3]=skew(fp_body[:,i]-R@self.pcb)
        bd=np.zeros(6)
        bd[0:3]=self.mass*(ddPcd-self.g)
        bd[3:6]=(R@self.Ib@R.T)@dWbd
        G=A.T@self.S@A+self.alpha*self.W+self.beta*self.U
        g0=-(bd@self.S@A+self.beta*self._Fp@self.U)
        def cost(F): return 0.5*F@G@F+g0@F
        def jac(F):  return G@F+g0
        bounds=[]; cons=[]
        for i in range(4):
            if contact[i]:
                bounds+=[(None,None),(None,None),(0,None)]
                k=3*i
                cons+=[{'type':'ineq','fun':lambda F,k=k: self.mu*F[k+2]-abs(F[k])},
                        {'type':'ineq','fun':lambda F,k=k: self.mu*F[k+2]-abs(F[k+1])},
                        {'type':'ineq','fun':lambda F,k=k+2: F[k]}]
            else:
                bounds+=[(0,0),(0,0),(0,0)]
        res=minimize(cost,self._Fp.copy(),jac=jac,method='SLSQP',
                     bounds=bounds,constraints=cons,
                     options={'ftol':1e-5,'maxiter':100,'disp':False})
        F=res.x; self._Fp=F.copy()
        out=np.zeros((3,4))
        for i in range(4): out[:,i]=F[3*i:3*i+3]
        return out

# ==============================================================================
# MuJoCoState – reads robot state from sim
# ==============================================================================

class MuJoCoState:
    def __init__(self, model, data):
        self.m=model; self.d=data
        self.qadr={}; self.dadr={}
        for k,name in enumerate(JOINT_NAMES):
            jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
            self.qadr[k]=model.jnt_qposadr[jid]
            self.dadr[k]=model.jnt_dofadr[jid]
        self.foot_ids=[
            mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)
            for n in ("FR_foot","FL_foot","RR_foot","RL_foot")]

    def get_q(self):
        q=np.zeros(12)
        for k in range(12): q[k]=self.d.qpos[self.qadr[k]]
        return q
    def get_qd(self):
        qd=np.zeros(12)
        for k in range(12): qd[k]=self.d.qvel[self.dadr[k]]
        return qd
    def get_body_pos(self):        return self.d.qpos[0:3].copy()
    def get_quat(self):            return self.d.qpos[3:7].copy()   # w,x,y,z
    def get_rot_mat(self):         return quat_to_rotmat_wxyz(self.get_quat())
    def get_yaw(self):
        R=self.get_rot_mat(); return math.atan2(R[1,0],R[0,0])
    def get_vel_global(self):      return self.d.qvel[0:3].copy()
    def get_gyro_body(self):       return self.d.qvel[3:6].copy()
    def get_gyro_global(self):     return self.get_rot_mat()@self.get_gyro_body()
    def get_feet_pos_global(self):
        fp=np.zeros((3,4))
        for i,bid in enumerate(self.foot_ids): fp[:,i]=self.d.xpos[bid]
        return fp

# ==============================================================================
# Actuator map
# ==============================================================================

def build_act_map(model):
    amap={}
    for k,aname in enumerate(ACTUATOR_NAMES):
        aid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,aname)
        amap[k]=aid
    return amap

# ==============================================================================
# Stand-up helper
# ==============================================================================

STAND_Q = np.array([0.0,0.9,-1.8]*4, dtype=float)

def stand_up(model, data, amap, state, viewer, duration=3.0, dt=0.002):
    steps=int(duration/dt)
    q0=state.get_q()
    for k in range(steps):
        a=k/steps
        qd=((1-a)*q0+a*STAND_Q)
        qc=state.get_q(); vc=state.get_qd()
        for j in range(12):
            if amap[j]>=0:
                data.ctrl[amap[j]]=50.0*(qd[j]-qc[j])+3.5*(0.0-vc[j])
        mujoco.mj_step(model,data)
        if k%10==0 and viewer is not None: viewer.sync()

# ==============================================================================
# Trotting controller  ←  State_Trotting.cpp
# ==============================================================================

class TrottingController:
    def __init__(self, vx=0.4, vy=0.0, wz=0.0, dt=0.002):
        self.rob=Go2Robot()
        self.wave=WaveGenerator(period=0.5,stance_ratio=0.5,
                                bias=np.array([0.0,0.5,0.5,0.0]))
        fc=FeetEndCal(self.wave.get_t_stance(), self.wave.get_t_swing())
        self.gait=GaitGenerator(self.wave, fc)
        self.bal=BalanceCtrl(self.rob)
        self.dt=dt

        # Gains (Go1 values from State_Trotting.cpp)
        self.Kpp=np.diag([70,70,70]); self.Kdp=np.diag([10,10,10])
        self.kpw=780.0;                self.Kdw=np.diag([70,70,70])
        self.KpSw=np.diag([400,400,400]); self.KdSw=np.diag([10,10,10])

        self.v_cmd=np.array([vx,vy,0.0])
        self.dyaw_cmd=wz; self.h=0.08

        self._pcd=None; self._yaw_cmd=None
        self._Rd=np.eye(3); self._wCG=np.zeros(3)
        self._init=False

    def step(self, s: MuJoCoState):
        q12=s.get_q(); qd12=s.get_qd()
        pos=s.get_body_pos(); vel=s.get_vel_global()
        R=s.get_rot_mat(); Ri=R.T
        yaw=s.get_yaw(); dyaw=s.get_gyro_global()[2]
        gyro_g=s.get_gyro_global()

        fp_b=self.rob.get_feet_pos_body(q12)
        fv_b=self.rob.get_feet_vel_body(q12,qd12)

        fp_g=np.zeros((3,4))
        fv_g=np.zeros((3,4))
        for i in range(4):
            fp_g[:,i]=pos+R@fp_b[:,i]
            fv_g[:,i]=vel+R@fv_b[:,i]

        try:    fp_g_mj=s.get_feet_pos_global()
        except: fp_g_mj=fp_g

        if not self._init:
            self._pcd=pos.copy(); self._pcd[2]=-FEET_NORMAL_STAND[2,0]
            self._yaw_cmd=yaw; self._Rd=rotz(yaw); self._init=True

        # Velocity cmd → global
        vcg=R@self.v_cmd
        vcg[0]=sat(vcg[0],vel[0]-0.2,vel[0]+0.2)
        vcg[1]=sat(vcg[1],vel[1]-0.2,vel[1]+0.2); vcg[2]=0.0
        self._pcd[0]=sat(self._pcd[0]+vcg[0]*self.dt,pos[0]-0.05,pos[0]+0.05)
        self._pcd[1]=sat(self._pcd[1]+vcg[1]*self.dt,pos[1]-0.05,pos[1]+0.05)
        self._yaw_cmd+=self.dyaw_cmd*self.dt
        self._Rd=rotz(self._yaw_cmd); self._wCG=np.array([0,0,self.dyaw_cmd])

        ph,ct=self.wave.calc()
        self.gait.set_gait(vcg[:2],self.dyaw_cmd,self.h)
        fp_goal,fv_goal=self.gait.run(fp_g_mj,pos,vel,yaw,dyaw,ph,ct)

        # Balance QP
        pe=self._pcd-pos; ve=vcg-vel
        ddP=np.clip(self.Kpp@pe+self.Kdp@ve,[-3,-3,-5],[3,3,5])
        dW=np.clip(self.kpw*rotmat_to_exp(self._Rd@Ri)+self.Kdw*(self._wCG-gyro_g),
                   [-40,-40,-10],[40,40,10])

        fp2B=np.zeros((3,4))
        for i in range(4): fp2B[:,i]=Ri@(fp_g_mj[:,i]-pos)

        Fg=-self.bal.cal_f(ddP,dW,R,fp2B,ct)

        for i in range(4):
            if ct[i]==0:
                Fg[:,i]=(self.KpSw@(fp_goal[:,i]-fp_g_mj[:,i])
                         +self.KdSw@(fv_goal[:,i]-fv_g[:,i]))

        Fb=np.zeros((3,4))
        for i in range(4): Fb[:,i]=Ri@Fg[:,i]
        tau=self.rob.get_tau(q12,Fb)

        # IK targets
        fpBg=np.zeros((3,4)); fvBg=np.zeros((3,4))
        for i in range(4):
            fpBg[:,i]=Ri@(fp_goal[:,i]-pos)
            fvBg[:,i]=Ri@(fv_goal[:,i]-vel)
        q_g=self.rob.get_q(fpBg)
        qd_g=self.rob.get_qd(fp2B,fvBg)

        kp=np.where(np.repeat(ct,3)==1, 10.0, 400.0)
        kd=np.where(np.repeat(ct,3)==1,  3.5,  10.0)
        return tau, q_g, qd_g, kp, kd

# ==============================================================================
# Main
# ==============================================================================

def main():
    xml_path = r"D:\Projects\Quadruped\unitree_mujoco\unitree_robots\go2\scene.xml"
    model=mujoco.MjModel.from_xml_path(xml_path)
    data =mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0)

    amap  = build_act_map(model)
    state = MuJoCoState(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Phase 1: Standing up …")
        stand_up(model, data, amap, state, viewer, duration=3.0)

        print("Phase 2: Trotting forward at 0.4 m/s …")
        ctrl = TrottingController(vx=0.4, vy=0.0, wz=0.0, dt=model.opt.timestep)
        step_count = 0

        while viewer.is_running():
            tau, qg, qdg, kp, kd = ctrl.step(state)
            qc=state.get_q(); vqc=state.get_qd()
            for j in range(12):
                if amap[j]>=0:
                    data.ctrl[amap[j]]=float(tau[j]+kp[j]*(qg[j]-qc[j])+kd[j]*(qdg[j]-vqc[j]))
            mujoco.mj_step(model, data)
            viewer.sync()
            step_count += 1

            # After 5 s → add a left turn
            if step_count == int(5.0/model.opt.timestep):
                print("Turning left (wz=0.3 rad/s) …")
                ctrl.dyaw_cmd = 0.3

if __name__=="__main__":
    main()
