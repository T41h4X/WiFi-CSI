# Project Overview: Wi-Fi CSI-Based Human Sensing & Activity Recognition

## 1. Introduction
Human Activity Recognition (HAR) has traditionally relied on wearable sensors (accelerometers, gyroscopes) or optical vision systems (RGB/depth cameras). While effective, wearables impose compliance burdens on subjects, and cameras suffer from severe privacy concerns, line-of-sight occlusions, and vulnerability to poor lighting conditions.

Wi-Fi Channel State Information (CSI) sensing provides a device-free, non-invasive, privacy-preserving alternative. By capturing how ambient radio frequency (RF) multipath propagation fluctuates when humans move through an environment, Wi-Fi systems can sense micro- and macro-movements without requiring cameras or wearables.

---

## 2. Core Physics: Why Wi-Fi CSI Senses Movement
In an indoor environment, Wi-Fi radio waves travel from a transmitter (AP) to a receiver (Raspberry Pi) across multiple reflection paths: bouncing off walls, floors, ceilings, furniture, and human bodies. This phenomenon is known as **multipath propagation**.

* **Received Signal Strength Indicator (RSSI)** provides only a coarse, scalar aggregate of total packet power. It is notoriously noisy and fails to capture fine-grained Doppler shifts or frequency-selective fading.
* **Channel State Information (CSI)** captures the physical-layer channel frequency response (CFR) across individual Orthogonal Frequency Division Multiplexing (OFDM) subcarriers:
  $$H(f, t) = |H(f, t)| e^{j \angle H(f, t)}$$
  where $|H(f, t)|$ represents subcarrier amplitude and $\angle H(f, t)$ represents phase.

When a human walks, sits, stands, or lies down, the length and attenuation of dynamic reflection paths change continuously. This introduces time-varying Doppler phase shifts and constructive/destructive interference patterns across distinct subcarrier frequencies, creating a unique RF "signature" for each activity.

---

## 3. High-Level System Architecture

```mermaid
flowchart LR
    A["Wi-Fi Access Point<br/>(KFTN, Channel 6)"] -->|"2.4 GHz RF Waves"| B["Human Motion in Environment"]
    B -->|"Multipath Propagation"| C["Raspberry Pi 5<br/>(BCM43455C0)"]
    C -->|"Nexmon CSI Driver"| D["UDP Port 5500"]
    D -->|"tcpdump"| E["PCAP Capture"]
    E -->|"Python Decoder"| F["CSI Dataset (CSV)"]
    F -->|"Standardization & Windowing"| G["Deep CNN Model"]
    G -->|"Inference"| H["Activity Class<br/>(Lying, Walking, Sitting, Standing)"]
```

---

## 4. Current Milestone & Status
This final-year development project (FYDP) has achieved two foundational objectives:
1. **Real-world physical acquisition**: Successfully extracted, streamed, and decoded 28,425 continuous 64-subcarrier CSI frames over UDP port 5500 on a Raspberry Pi 5 under Linux 13 ARM64.
2. **Deep learning recognition pipeline**: Implemented, standardized, and rigorously validated a 93,636-parameter Convolutional Neural Network across 5 random seeds on a 4-class human activity dataset, reaching **94.37% peak accuracy** (and **91.41% ± 2.42%** multi-seed mean accuracy).
