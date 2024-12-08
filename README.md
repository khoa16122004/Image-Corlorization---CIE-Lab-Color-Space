# Image-Corlorization with CIE-Lab ColorSpace
Exploring Different Machine Learning Strategies for Tackling the Image Colorization Problem

This project was inspired by this paper: [Colorful Image Colorization](https://arxiv.org/abs/1603.08511)

![alt text](figure/result.png)

# Usage
- For inference with specific image path
    ```bash    
    python inference.py --checkpoint <checkpoint_path> --outdir <output_directory> --dataset <dataset_name>
    ```
- For training the model
    ```bash
    python train.py --epochs <num_epochs> --checkpoint <checkpoint_path> --checkpoint-G <checkpoint_generator_path> --checkpoint-D <checkpoint_discriminator_path> --lambda_gp <lambda_gp> --n_critic <n_critic> --lr <lr> --step_size <step_size> --gamma <gamma> --colorizer <colorizer_type> --batch_size <batch_size> --objective <objective_function> --arch <model_architecture> --outdir <output_directory> --dataset <dataset_name>
    ```

- For testing the model
    ```bash
    python test.py --checkpoint <checkpoint_path> --outdir <output_directory> --dataset <dataset_name>
    ```

# Image Reconstruction Approach
- Reason why using CIE-Lab color space
![alt text](figure/limit_rgb.png)

- Pipeline
![alt text](figure/recons_pipeline.png)

# Image Classification Approach
- Reason why using CIE-Lab color space
![alt text](figure/limit_rgb_.png)

- Pipeline
![alt text](figure/classification_pipeline.png)

# GAN Approach
- Gan Pipeline approach
![alt text](figure/gan.png)

# Architecture
- Reconstruction/Classification approach
    ![alt text](figure/arch_1.png)

- GAN approach
    ![alt text](figure/arch_2.png)

