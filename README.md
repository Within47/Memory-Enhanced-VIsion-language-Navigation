# Memory-Enhanced-VIsion-language-Navigation

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
## 2. Setup and Configuration Guide


### 2.1 Environment Setup

The project has been tested on **Ubuntu 22.04**, and compatibility with other systems has not yet been validated.  
The VLFM project requires the following core environments and dependencies:

1. Python 3.8+
2. PyTorch 1.10+
3. NumPy, SciPy, OpenCV
4. Habitat-Sim and Habitat-Lab (for simulation environments)

#### Installation Steps:

1. Clone the project repository:  
   [https://github.com/bdaiinstitute/vlfm](https://github.com/bdaiinstitute/vlfm)

2. Create a new conda environment:  
   ```bash
   conda create -n MM_UA python=3.8
### 2.2 How to Run the System
Script path: scripts/run_vlfm_test.sh
Usage:
bash ./scripts/run_vlfm_test.sh [MODE] [MAX_STEPS]

Example: Run full mode with a maximum of 500 steps
bash ./scripts/run_vlfm_test.sh MM-Nav_UA-Nav 500

Example: Run MM-Nav only with 300 steps
bash ./scripts/run_vlfm_test.sh MM-Nav 300

