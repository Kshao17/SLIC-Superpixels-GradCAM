import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.segmentation import slic as official_slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

class SLIC_Implementation:
    def __init__(self, image, K, m=10):
        self.image = color.rgb2lab(image) 
        self.height, self.width = self.image.shape[:2]
        self.K = K
        self.m = m
        self.N = self.height * self.width
        self.S = int(np.sqrt(self.N / self.K)) 
        
        self.centers = self._init_centers()
        self.distances = np.full((self.height, self.width), np.inf)
        self.labels = np.full((self.height, self.width), -1)

    def _init_centers(self):
        centers = []
        for i in range(self.S // 2, self.height, self.S):
            for j in range(self.S // 2, self.width, self.S):
                centers.append(self._get_center(i, j))
        return np.array(centers)

    def _get_center(self, i, j):
        return np.array([*self.image[i, j], i, j])

    def iterate(self, num_iterations=10):
        for _ in range(num_iterations):
            self.distances.fill(np.inf)
            for idx, center in enumerate(self.centers):
                y_min = max(int(center[3] - self.S), 0)
                y_max = min(int(center[3] + self.S), self.height)
                x_min = max(int(center[4] - self.S), 0)
                x_max = min(int(center[4] + self.S), self.width)

                crop = self.image[y_min:y_max, x_min:x_max]
                
                d_color = np.sqrt(np.sum((crop - center[:3])**2, axis=2))
                yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
                d_space = np.sqrt((yy - center[3])**2 + (xx - center[4])**2)
                
                D = np.sqrt(d_color**2 + (self.m / self.S)**2 * d_space**2)
                
                mask = D < self.distances[y_min:y_max, x_min:x_max]
                self.distances[y_min:y_max, x_min:x_max][mask] = D[mask]
                self.labels[y_min:y_max, x_min:x_max][mask] = idx

            for idx in range(len(self.centers)):
                mask = (self.labels == idx)
                if np.any(mask):
                    self.centers[idx][:3] = np.mean(self.image[mask], axis=0)
                    coords = np.argwhere(mask)
                    self.centers[idx][3:] = np.mean(coords, axis=0)

    def get_boundaries(self):
        from skimage.segmentation import mark_boundaries
        return mark_boundaries(color.lab2rgb(self.image), self.labels)

if __name__ == "__main__":
    img = io.imread('brandeis_campus.jpg') 
    img = resize(img, (300, 400), anti_aliasing=True)
    
    for k_val in [50, 200, 500]:   
        slic_obj = SLIC_Implementation(img, K=k_val, m=10)
        slic_obj.iterate(num_iterations=10)
        res_my = slic_obj.get_boundaries() 
        
        labels_official = official_slic(img, n_segments=k_val, compactness=10, start_label=1)
        res_official = mark_boundaries(img, labels_official)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        ax1.imshow(res_my)
        ax1.set_title(f'My Implementation (K={k_val})')
        ax1.axis('off')
        
        ax2.imshow(res_official)
        ax2.set_title(f'Scikit-image SLIC (K={k_val})')
        ax2.axis('off')
        
        plt.savefig(f"comparison_K_{k_val}.png")
        plt.show()