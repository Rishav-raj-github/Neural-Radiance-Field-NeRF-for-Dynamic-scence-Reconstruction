# Neural-Radiance-Field-NeRF-for-Dynamic-scence-Reconstruction
Real Time Neural Radiance Field(NeRF) for Dynamic scence Reconstruction on Edge GPUs.



Real-Time Neural Radiance Field (NeRF) for Dynamic Scene Reconstruction on Edge GPUs

A research-driven engineering project focused on bringing Neural Radiance Fields (NeRF) to life — in **real time**, on **edge GPUs**, and in **dynamic, changing environments**.

This project tackles the key challenge: NeRFs are powerful, but they’re heavy. So how do we run them fast enough — and smart enough — on devices with limited computing power like Jetson Nano, Xavier, or mobile GPUs?

---

What is this project about?

Traditional NeRFs are great at capturing beautiful 3D reconstructions from static scenes using just a few camera views. But most of them need powerful GPUs and take minutes to hours to render. That’s fine for research or offline rendering, but it doesn’t cut it when you're building **real-world, real-time applications** like:

- AR/VR apps on mobile or wearables
- Real-time scene understanding in robotics
- Smart surveillance with 3D modeling
- Real-time simulation for edge-based metaverse tools

This project is my attempt to re-engineer NeRF — optimizing it to **reconstruct dynamic scenes on the fly** using lightweight models that run on **edge GPUs** in **real time**.

---

Key Objectives

- 🧠 Rebuild NeRF for **speed and adaptability**
- 🔁 Extend it to work on **dynamic (non-static) scenes**
- ⚡ Optimize for **real-time inference** on edge devices (like NVIDIA Jetson)
- 📦 Keep memory and compute usage as low as possible
- 🤖 Enable seamless integration with real-time sensors (like RGB-D or stereo cams)

---

What Makes This Different?

✅ Most NeRFs assume the world doesn’t move — this one doesn’t  
✅ Most require cloud GPUs — this runs **on-device**  
✅ Most are slow — this is built to be **real-time**  
✅ Most models are bulky — this project **compresses and quantizes**

---

## 🔬 Core Techniques & Architecture

This system borrows ideas from several cutting-edge NeRF variants, including:

| Variant          | Used For |
|------------------|----------|
| Instant-NGP      | Lightning-fast NeRF backbone with hash encodings  
| KiloNeRF         | Neural network field splitting for speed on parallel devices  
| DynamicNeRF      | Scene modeling with temporal dynamics  
| FastNeRF         | For efficient forward rendering and precomputation  
| NeRF-Tex, NeRFies | For scene realism and motion modeling  

These techniques are merged, trimmed, and adapted to fit within the compute envelope of an edge GPU while still reconstructing scenes with acceptable fidelity.

---

High-Level Architecture

