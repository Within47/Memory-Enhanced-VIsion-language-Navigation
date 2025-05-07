# Memory-Enhanced-Vision-Language-Navigation

---

## Table of Contents

- [1. Project Overview: A New Paradigm of Cognitive Navigation](#1-project-overview-a-new-paradigm-of-cognitive-navigation)
- [2. Setup and Configuration Guide](#2-setup-and-configuration-guide)

---

## 1. Project Overview: A New Paradigm of Cognitive Navigation

The **VLFM (Vision-Language Frontier Map)** project is an advanced intelligent navigation system that integrates visual understanding, language processing, and spatial planning to enable efficient goal-oriented exploration in unknown environments. Building upon the original VLFM framework ([bdaiinstitute/vlfm](https://github.com/bdaiinstitute/vlfm.git)), this project introduces an innovative **"Think–Memory–Action" framework**, reconstructing the foundation of intelligent navigation from a cognitive science perspective.

### Core Challenges and Solutions

The VLFM project addresses several critical limitations in traditional vision-language navigation approaches:

1. **Lack of Long-Term Memory**  
   Traditional methods fail to leverage historical observations effectively, resulting in redundant exploration and cyclic behaviors.  
   - **Solution**: *Memory-Augmented Navigation (MM-Nav)* with structured spatial memory and repulsion mechanisms to enhance historical trajectory awareness and reduce revisits.

2. **Poor Environmental Adaptability**  
   Conventional navigation strategies struggle to adapt to varying environmental complexities.  
   - **Solution**: *Uncertainty-Aware Adaptive Navigation (UA-Nav)* dynamically adjusts exploration strategies based on cognitive, environmental, and navigational uncertainties.

---

## 2. Setup and Configuration Guide

This section provides a step-by-step guide to help you set up the development environment and run the **Memory-Enhanced Vision-Language Navigation System** smoothly.

### 2.1 Environment Setup

> ✅ *Tested on: **Ubuntu 22.04 LTS***  
> ⚠️ *Compatibility with other operating systems is not guaranteed.*

#### Required Dependencies

Ensure the following software environments and libraries are installed:

- Python **3.8+**
- PyTorch **1.10+**
- **NumPy**, **SciPy**, **OpenCV**
- **Habitat-Sim** and **Habitat-Lab** *(for 3D simulation and rendering)*

#### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bdaiinstitute/vlfm.git
   cd vlfm

2. Create a new conda environment:  
   ```bash
   conda create -n MM_UA python=3.8
### 2.2 How to Run the System
Script path:```bash scripts/run_vlfm_test.sh
Usage:
   ```bash
   bash ./scripts/run_vlfm_test.sh [MODE] [MAX_STEPS]
   ```
Example: Run full mode with a maximum of 500 steps

   ```bash
   bash ./scripts/run_vlfm_test.sh MM-Nav_UA-Nav 500
   ```
Example: Run MM-Nav only with 300 steps

   ```bash
   bash ./scripts/run_vlfm_test.sh MM-Nav 300
   ```

## 3. Final Results and demo
| Method                     | SR (%) ↑ | SPL ↑   | Yaw (°) ↓ |
|----------------------------|----------|---------|-----------|
| Baseline                   | 52.40    | 0.3036  | 2.40      |
| Only Memory-Augmention     | 52.30    | 0.3066  | 5.03      |
| Only Uncertainty-Awareness | 52.70    | 0.3065  | 3.03      |
| **Full version**           | **53.36**| **0.3193** | 3.26      |
[Demo Video](./demo.mp4)
