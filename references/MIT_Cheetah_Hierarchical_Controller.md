# Hierarchical controller for highly dynamic locomotion utilizing pattern modulation and impedance control : implementation on the MIT Cheetah robot

**Author:** Hae-Won Park (MIT Biomimetic Robotics Lab)
**Year:** 2013-2015
**Institution:** Massachusetts Institute of Technology

## Download Link
Due to AWS WAF (Web Application Firewall) protections on MIT's DSpace repository, automated downloads are frequently blocked with a Captcha challenge. 

Please download the PDF manually from the link below and save it as `MIT_Cheetah_Hierarchical_Controller.pdf` in this `references/` folder.

**Paper URL:** [https://dspace.mit.edu/handle/1721.1/85490](https://dspace.mit.edu/handle/1721.1/85490)
**Direct PDF Link:** [https://dspace.mit.edu/bitstreams/9ee27d50-b481-4644-aadf-db83b6c3f192/download](https://dspace.mit.edu/bitstreams/9ee27d50-b481-4644-aadf-db83b6c3f192/download)

## Relevance to Quadruped Project
This is the foundational thesis that the `chvmp/champ` framework's kinematics and locomotion controller is based on. It introduces:
1. Pattern modulation for gait generation (Trot, Pace, Bound) using an internal clock layer.
2. Generating 3D Cartesian foot trajectories (swing profiles).
3. Using Impedance Control and Inverse Kinematics to calculate the required actuator torques and angles.
