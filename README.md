# Denoise SPECT 2D images with Deep Image Prior
Deep Image Prior (DIP) was discribed in:

<a href="http://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf">Ulyanov, Dmitry, et al. "Deep Image Prior." The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 9446-9454.</a>

This is an application of DIP in image denoising in reconstructed 2D SPECT images.

## Sample Results:
![](https://github.com/junyuchen245/SPECT-Img-Denoising-DIP/blob/master/sample_img/var001.gif)

For simplest testing, input noisy image is a 2D slice of a 3D SPECT image with the additive Gaussian noise (Var=0.001, 0.005, and 0.01).
#### Variance = 0.001:
![](https://github.com/junyuchen245/SPECT-Img-Denoising-DIP/blob/master/sample_img/var=001.png)
#### Variance = 0.005:
![](https://github.com/junyuchen245/SPECT-Img-Denoising-DIP/blob/master/sample_img/var=005.png)
#### Variance = 0.01:
![](https://github.com/junyuchen245/SPECT-Img-Denoising-DIP/blob/master/sample_img/var=01.png)
