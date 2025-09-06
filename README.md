![FocusLab](https://github.com/IcedWatermelonJuice/IcedWatermelonJuice/blob/main/FocusLab_Logo.png?raw=true)

# 📢 FS-SEI Method Based on P3MC
* The code corresponds to our **early access** paper "[P3MC: Dual-Level Data Augmentation for Robust Few-Shot Specific Emitter Identification](https://ieeexplore.ieee.org/document/11073147)".
* If you find our project helpful for your research, please consider citing our work. Thank you!
```
@ARTICLE{11073147,
  author={Xu, Lai and Zhang, Weijie and Tang, Tiantian and Zhang, Qianyun and Lin, Yun and Xuan, Qi and Gui, Guan},
  journal={IEEE Internet of Things Journal}, 
  title={P3MC: Dual-Level Data Augmentation for Robust Few-Shot Specific Emitter Identification}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Data augmentation;Feature extraction;Internet of Things;Transformers;Security;Authentication;Transfer learning;Time series analysis;Generative adversarial networks;Training;Specific emitter identification (SEI);few-shot learning;self-supervised learning;data augmentation;feature augmentation},
  doi={10.1109/JIOT.2025.3586923}}
```

# 📑 Requirement
![python 3.8](https://img.shields.io/badge/python-3.8-blue)
![pytorch 1.10.2](https://img.shields.io/badge/pytorch-1.10.2-blue)
![numpy 1.21.5](https://img.shields.io/badge/numpy-1.21.5-blue)
![pillow 9.4.0](https://img.shields.io/badge/pillow-9.4.0-blue)
![tftb](https://img.shields.io/badge/tftb-9.4.0-blue)
![scikit-learn 1.0.2](https://img.shields.io/badge/scikit--learn-1.0.2-blue)
![tqdm 4.64.1](https://img.shields.io/badge/tqdm-4.64.1-blue)
![pandas 1.3.5](https://img.shields.io/badge/pandas-1.3.5-blue)

# 🎯 Deployment
### Preparation
To run this project locally, please follow these steps:
1. **Clone this repository**:
   ```bash
   git clone https://github.com/IcedWatermelonJuice/P3MC.git
   cd P3MC
   ```
2. **Prepare the dataset directory**:  
   The `Datasets/` folder and its subdirectories are not included by default.  
   You need to manually create the following directory structure:
   ```
   Datasets/
   ├── ADS-B/
   ├── WiFi_ft62/
   └── LoRa/
   ```

3. **Download the datasets**:  
   Please refer to the **Datasets** section for download links to the datasets.

### Project Structure
- **`models/`**: Contains model architectures and loss function implementations.
- **`utils/`**: Includes custom utility functions required to run the code.
- **`runs/`**: Stores training logs, model checkpoints, and experimental results.  
  We have released the logs and model weights corresponding to the results reported in the paper.
### Key Scripts
- **`pretext.py`**: Used during the pre-training stage.  
  For example, you can run the following command to obtain a pretrained encoder:  
  ```bash
  CUDA_VISIBLE_DEVICES=0 python pretext.py -e CVTSLANet
  ```
- **`downstream.py`**: Used for downstream few-shot SEI tasks.  
  To fine-tune the encoder and train a classifier for a 5-shot scenario, run:  
  ```bash
  CUDA_VISIBLE_DEVICES=0 python downstream.py -e CVTSLANet -c lr -s 5
  ```  
  To evaluate under artificially added noise with a specified SNR (e.g., 10 dB) during the test phase, use:  
  ```bash
  CUDA_VISIBLE_DEVICES=0 python downstream.py -e CVTSLANet -c lr -s 5 --snr_enable --snr 10
  ```
> **Note**: If you have already specified the GPU device within your Python scripts using `os.environ['CUDA_VISIBLE_DEVICES']`, or if you do not need to manually assign a GPU,  you can safely remove the `CUDA_VISIBLE_DEVICES=...` prefix from the above commands.

# 💽Datasets
### 1. ADS-B
- **Raw Dataset**  
  [https://gitee.com/heu-linyun/Codes](https://gitee.com/heu-linyun/Codes)
- **Preprocessed Dataset and Scripts**  
  [https://cloud.xulai.work/Datasets/ADS-B](https://cloud.xulai.work/Datasets/ADS-B)

### 2. Wi-Fi (62 feet)
- **Raw Dataset**  
  [https://genesys-lab.org/oracle](https://genesys-lab.org/oracle)
- **Preprocessed Dataset and Scripts**  
  [https://cloud.xulai.work/Datasets/WiFi_ft62](https://cloud.xulai.work/Datasets/WiFi_ft62)

### 3. LoRa
- **Raw Dataset**  
  [https://ieee-dataport.org/open-access/lorarffidataset](https://ieee-dataport.org/open-access/lorarffidataset)
- **Preprocessed Dataset and Scripts**  
  [https://cloud.xulai.work/Datasets/LoRa](https://cloud.xulai.work/Datasets/LoRa)

# ✉️ Contact
* E-mail: [2025011313@njupt.edu.cn](mailto:2025011313@njupt.edu.cn).
* Issues: [GitHub Issues](https://github.com/IcedWatermelonJuice/P3MC/issues).
