from tqdm import tqdm 
import os 
import numpy as np 
from tqdm import tqdm 
import cv2 
from sklearn.mixture import GaussianMixture as GMM

PATH = "/Dataset" 

training_img_path = "INPUT_IMG" 
training_mask_path = "GT" 

gmm = GMM(n_components=2, covariance_type="tied") 

training_img = [] 
mask_img = [] 
gmm_seg_images = []
# fft_img = []

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


for img_name in tqdm(os.listdir(os.path.join(PATH, training_img_path))):  
    img = cv2.imread(os.path.join(PATH, training_img_path, img_name)) 
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST) 
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    seg_name = img_name.split(".")[0] + "_segmentation.png" 
    seg_img = cv2.imread(os.path.join(PATH, training_mask_path, seg_name), cv2.IMREAD_GRAYSCALE) 
    seg_img = cv2.resize(seg_img, (256, 256), interpolation= cv2.INTER_NEAREST)
    
    gmm_model = gmm.fit(img.reshape((-1, 3))) 
    pdf = np.exp(gmm.score_samples(img.reshape((-1, 3))))
    
    # fft_image = np.fft.fft2(img)
    # fft_shifted = np.fft.fftshift(fft_image)
    # magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))
    
    gmm_seg_images.append(pdf.reshape(img.shape[0], img.shape[1])) 
    # fft_img.append(magnitude_spectrum)
    training_img.append(img/255.) 
    mask_img.append(seg_img/255.)


training_img_np = np.array(training_img) 
mask_img_np = np.array(mask_img)
gmm_seg_images = np.array(gmm_seg_images)
# fft_img = np.array(fft_img) 

print(training_img_np.shape, mask_img_np.shape, gmm_seg_images.shape)


np.savez("prepared_data.npz", images=training_img_np, masks=mask_img_np, gmm_seg_images=gmm_seg_images)