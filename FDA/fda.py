import numpy as np
import random
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import torchvision.transforms as T

##############################################################################################################################################################################

class StyleAugment:

    def __init__(self, n_images_per_style=10, L=0.1, size=(1024, 512)):
        self.styles = []
        self.styles_names = []
        self.n_images_per_style = n_images_per_style
        self.L = L
        self.size = size         
        self.sizes = None
        self.cv2 = False
    
    def preprocess(self, x):
        x = x.resize(self.size, Image.BICUBIC)
        x = np.asarray(x, np.float32)
        x = x[:, :, ::-1]
        x = x.transpose((2, 0, 1))
        return x.copy()
    
    def deprocess(self, x, size):
        x = Image.fromarray(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1])
        x = x.resize(size, Image.BICUBIC)
        return x

    def add_style(self, dataset, name=None):
        
        dataset.return_PIL = True

        if name is not None:
            self.styles_names.append(name)

        n = 0
        styles = []
        
        for sample, _ in tqdm(dataset, total=min(len(dataset), self.n_images_per_style) if self.n_images_per_style > 0 else len(dataset)):
            image = self.preprocess(sample)
            if n >= self.n_images_per_style and self.n_images_per_style > 0:
                break
            styles.append(self._extract_style(image))
            n += 1

        if self.n_images_per_style != 1:
            styles = np.stack(styles, axis=0)
            style = np.mean(styles, axis=0)
            self.styles.append(style)
        else:
            self.styles += styles
            
        dataset.return_PIL = False

    def _extract_style(self, img_np):
        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp = np.abs(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        if self.sizes is None:
            self.sizes = self.compute_size(amp_shift)
        h1, h2, w1, w2 = self.sizes
        style = amp_shift[:, h1:h2, w1:w2]
        return style

    def compute_size(self, amp_shift):
        _, h, w = amp_shift.shape
        b = (np.floor(np.amin((h, w)) * self.L)).astype(int) 
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)
        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        return h1, h2, w1, w2
    
    def apply_style(self, image):
        return self._apply_style(image)

    def _apply_style(self, img):

        if self.n_images_per_style == 0:
            return img

        n = random.randint(0, len(self.styles) - 1)
        style = self.styles[n]

        W, H = img.size
        img_np = self.preprocess(img)

        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp, pha = np.abs(fft_np), np.angle(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        amp_shift[:, h1:h2, w1:w2] = style
        amp_ = np.fft.ifftshift(amp_shift, axes=(-2, -1))

        fft_ = amp_ * np.exp(1j * pha)
        img_np_ = np.fft.ifft2(fft_, axes=(-2, -1))
        img_np_ = np.real(img_np_)
        img_np__ = np.clip(np.round(img_np_), 0., 255.)

        img_with_style = self.deprocess(img_np__, (W, H))

        return img_with_style
    

class StyleAugmentClustered:

    def __init__(self, args, size=(1024, 512)):
        self.styles = []
        self.styles_names = []
        self.n_images_per_style = args.num_images_per_style
        self.L = args.L
        self.size = size        
        self.sizes = None
        self.cv2 = False
        self.current_cluster = None   
        self.args = args
        self.toPIL = T.ToPILImage()                               
    
    def preprocess(self, x):
        x = x.resize(self.size, Image.BICUBIC)
        x = np.asarray(x, np.float32)
        x = x[:, :, ::-1]
        x = x.transpose((2, 0, 1))
        return x.copy()
    
    def deprocess(self, x, size):
        x = Image.fromarray(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1])
        x = x.resize(size, Image.BICUBIC)
        return x

    def add_style(self, dataset, name=None):

        dataset.return_PIL = True
        
        if name is not None:
            self.styles_names.append(name)

        n = 0
        styles = []
        
        for sample, _ in tqdm(dataset, total=min(len(dataset), self.n_images_per_style) if self.n_images_per_style > 0 else len(dataset)):
            image = self.preprocess(sample)
            if n >= self.n_images_per_style and self.n_images_per_style > 0:
                break
            styles.append(self._extract_style(image))
            n += 1

        if self.n_images_per_style != 1:
            styles = np.stack(styles, axis=0)
            style = np.mean(styles, axis=0)
            self.styles.append(style)
        else:
            self.styles += styles

        dataset.return_PIL = False

    def _extract_style(self, img_np):
        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp = np.abs(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        if self.sizes is None:
            self.sizes = self.compute_size(amp_shift)
        h1, h2, w1, w2 = self.sizes
        style = amp_shift[:, h1:h2, w1:w2]
        return style

    def extract_style_for_test(self,img_tensor):
      img = self.toPIL(img_tensor.squeeze(0))
      img = self.preprocess(img)
      return self._extract_style(img)
        
    def create_clustering(self):
        
        self.cluster_mapping = dict()                   #Dict used to map the clients to each cluster of the best clustering. 
        self.cluster_styles_mapping =  dict()           #Dict used to map, for each cluster_id, the indexes (in variable "styles") of the styles belonging to the cluster
        clustering_list = list()                        #List used to store all the clusterings. [NOTE: Only the best clustering will be selected according to the silhouette score]
        res_list = list()                               #List used to store the predicted clusters labels. 
        score_list = list()                             #List used to store the silhoutte scores of the clusterings                

        styles_flat = np.array(self.styles).reshape(len(self.styles), -1)

        m, n = self.args.m_cl , self.args.n_cl
        k_list = list(range(m,n))
        N = self.args.N_cl
        
        print("\nStarting Clustering")
        for k in k_list:
            clustering = KMeans(n_clusters = k, n_init = N).fit(styles_flat)
            clustering_list.append(clustering)
            res_list.append(clustering.labels_)
            score_list.append(silhouette_score(styles_flat, clustering.labels_))
        print("\nFinished!")

        best_id = np.argmax(score_list)
        self.best_clustering = clustering_list[best_id]
        self.styles_labels = res_list[best_id]               #cluster labels of the styles in the best clustering: [0, 2, 1, 3 ,0 ,2, 1,5 ,...]
        best_k = k_list[best_id]
        print("\nBest k: ",best_k)  

        # Mapping the clients to each cluster of the best clustering: cluster_mapping = {0: [c_id1,c_id4], 1: [c_id2, ...], ....}
        for cluster_id in range(best_k):
            self.cluster_mapping[cluster_id] = [self.styles_names[i] for i,cluster_label in enumerate(res_list[best_id]) if cluster_label == cluster_id]

        # Mapping the styles to each cluster of the best clustering: cluster_styles_mapping {0: [idx_style_cid1, idx_style_cid4], 1: [idx_style_cid2, ...], .... }
        for k,v in self.cluster_mapping.items():
            self.cluster_styles_mapping[k] = [self.styles_names.index(item) for item in v]

        self.plot_clusters(styles_flat)

    def plot_clusters(self, styles_flat):
        
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_data = np.array(np.append(styles_flat, self.best_clustering.cluster_centers_,axis=0))
        tsne_data = tsne.fit_transform(tsne_data)

        pca = PCA(n_components = 2, random_state = 42)
        pca_data = pca.fit_transform(styles_flat)
        pca_cluster_centers = pca.transform(self.best_clustering.cluster_centers_)

        n_clusters = len(self.best_clustering.cluster_centers_)
        plt.scatter(tsne_data[:-n_clusters, 0], tsne_data[:-n_clusters, 1], c= self.best_clustering.labels_, cmap='viridis')
        plt.scatter(tsne_data[-n_clusters:, 0], tsne_data[-n_clusters:, 1], marker='x', color='red', label='Cluster centers')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('KMeans Clustering in t-SNE space (2D)')
        plt.legend()
        os.makedirs(os.path.join("Results", "Clustering"), exist_ok=True)
        plt.savefig(os.path.join("Results", "Clustering", "tSNE_5.1_"+str(self.L)+".png"))
        plt.close()

        plt.scatter(pca_data[:, 0], pca_data[:, 1], c= self.best_clustering.labels_, cmap='viridis')
        plt.scatter(pca_cluster_centers[:, 0], pca_cluster_centers[:, 1], marker='x', color='red', label='Cluster centers')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('KMeans Clustering in PCA space (2D)')
        plt.legend()
        os.makedirs(os.path.join("Results", "Clustering"), exist_ok=True)
        plt.savefig(os.path.join("Results", "Clustering", "PCA_5.1_"+str(self.L)+".png")) 
        plt.close()

    def compute_size(self, amp_shift):
        _, h, w = amp_shift.shape
        b = (np.floor(np.amin((h, w)) * self.L)).astype(int) 
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)
        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        return h1, h2, w1, w2
    
    def apply_style(self, image):
        return self._apply_style(image)

    def _apply_style(self, img):

        if self.n_images_per_style == 0:
            return img

        n = random.choice(self.cluster_styles_mapping[self.current_cluster])
        style = self.styles[n]
        
        W, H = img.size
        img_np = self.preprocess(img)

        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp, pha = np.abs(fft_np), np.angle(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        amp_shift[:, h1:h2, w1:w2] = style
        amp_ = np.fft.ifftshift(amp_shift, axes=(-2, -1))

        fft_ = amp_ * np.exp(1j * pha)
        img_np_ = np.fft.ifft2(fft_, axes=(-2, -1))
        img_np_ = np.real(img_np_)
        img_np__ = np.clip(np.round(img_np_), 0., 255.)

        img_with_style = self.deprocess(img_np__, (W, H))

        return img_with_style
    

class StyleAugmentExtended:

    def __init__(self, args, size=(1024, 512)):
        self.styles = []
        self.styles_names = []
        self.n_images_per_style = args.num_images_per_style
        self.L = args.L
        self.mode = args.mode
        self.p_interpolation = args.p_interpolation
        self.size = size        
        self.sizes = None
        self.cv2 = False
        self.current_cluster = None   
        self.args = args
        self.toPIL = T.ToPILImage()                               
    
    def preprocess(self, x):
        if isinstance(x, np.ndarray):         
            x = cv2.resize(x, self.size, interpolation=cv2.INTER_CUBIC)
            self.cv2 = True
        else:
            x = x.resize(self.size, Image.BICUBIC)
        x = np.asarray(x, np.float32)
        x = x[:, :, ::-1]
        x = x.transpose((2, 0, 1))
        return x.copy()
    
    def deprocess(self, x, size):
        if self.cv2:
            x = cv2.resize(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1], size, interpolation=cv2.INTER_CUBIC)
        else:
            x = Image.fromarray(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1])
            x = x.resize(size, Image.BICUBIC)
        return x

    def add_style(self, dataset, name=None):
        
        dataset.return_PIL = True

        if name is not None:
            self.styles_names.append(name)

        n = 0
        styles = []
        
        for sample, _ in tqdm(dataset, total=min(len(dataset), self.n_images_per_style) if self.n_images_per_style > 0 else len(dataset)):
            image = self.preprocess(sample)
            if n >= self.n_images_per_style and self.n_images_per_style > 0:
                break
            styles.append(self._extract_style(image))
            n += 1

        if self.n_images_per_style != 1:
            styles = np.stack(styles, axis=0)
            style = np.mean(styles, axis=0)
            self.styles.append(style)
        else:
            self.styles += styles

        dataset.return_PIL = False

    def _extract_style(self, img_np):
        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp = np.abs(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        if self.sizes is None:
            self.sizes = self.compute_size(amp_shift)
        h1, h2, w1, w2 = self.sizes
        style = amp_shift[:, h1:h2, w1:w2]
        return style

    def create_bank_clustered(self):
        
        self.cluster_mapping = dict()               #Dict used to map the clients to each cluster of the best clustering. 
        self.cluster_styles_mapping =  dict()       #Dict used to map, for each cluster_id, the indexes (in variable "styles") of the styles belonging to the cluster
        clustering_list = list()                    #List used to store all the clusterings. [NOTE: Only the best clustering will be selected according to the silhouette score]
        res_list = list()                           #List used to store the predicted clusters labels. 
        score_list = list()                         #List used to store the silhoutte scores of the clusterings

        styles_flat = np.array(self.styles).reshape(len(self.styles), -1)

        m, n = self.args.m_cl , self.args.n_cl
        k_list = list(range(m,n))
        N = self.args.N_cl
        
        print("\nStarting Clustering")
        for k in k_list:
            clustering = KMeans(n_clusters = k, n_init = N).fit(styles_flat)
            clustering_list.append(clustering)
            res_list.append(clustering.labels_)
            score_list.append(silhouette_score(styles_flat, clustering.labels_))
        print("\nFinished!")

        best_id = np.argmax(score_list)
        self.best_clustering = clustering_list[best_id]
        self.styles_labels = res_list[best_id]                      #cluster labels of the styles in the best clustering: [0, 2, 1, 3 ,0 ,2, 1,5 ,...]
        best_k = k_list[best_id]
        print("\nBest k: ",best_k)      

        # Mapping the clients to each cluster of the best clustering: cluster_mapping = {0: [c_id1,c_id4], 1: [c_id2, ...], ....}
        for cluster_id in range(best_k):
            self.cluster_mapping[cluster_id] = [self.styles_names[i] for i,cluster_label in enumerate(res_list[best_id]) if cluster_label == cluster_id]

        # Mapping the styles to each cluster of the best clustering: cluster_styles_mapping {0: [idx_style_cid1, idx_style_cid4], 1: [idx_style_cid2, ...], .... }
        for k,v in self.cluster_mapping.items():
            self.cluster_styles_mapping[k] = [self.styles_names.index(item) for item in v]

        self.plot_clusters(styles_flat)

        styles_bank = []
        for style_ids in self.cluster_styles_mapping.values():
            styles = [self.styles[i] for i in style_ids]
            styles = np.stack(styles, axis=0)
            style = np.mean(styles, axis=0)
            styles_bank.append(style)
        self.styles = styles_bank

    def plot_clusters(self, styles_flat):
        
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        tsne_data = np.array(np.append(styles_flat, self.best_clustering.cluster_centers_,axis=0))
        tsne_data = tsne.fit_transform(tsne_data)

        pca = PCA(n_components = 2, random_state = 42)
        pca_data = pca.fit_transform(styles_flat)
        pca_cluster_centers = pca.transform(self.best_clustering.cluster_centers_)

        n_clusters = len(self.best_clustering.cluster_centers_)
        plt.scatter(tsne_data[:-n_clusters, 0], tsne_data[:-n_clusters, 1], c= self.best_clustering.labels_, cmap='viridis')
        plt.scatter(tsne_data[-n_clusters:, 0], tsne_data[-n_clusters:, 1], marker='x', color='red', label='Cluster centers')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('KMeans Clustering in t-SNE space (2D)')
        plt.legend()
        os.makedirs(os.path.join("Results", "Clustering"), exist_ok=True)
        plt.savefig(os.path.join("Results", "Clustering", "tSNE_ext2.png"))

        plt.close()

        plt.scatter(pca_data[:, 0], pca_data[:, 1], c= self.best_clustering.labels_, cmap='viridis')
        plt.scatter(pca_cluster_centers[:, 0], pca_cluster_centers[:, 1], marker='x', color='red', label='Cluster centers')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.title('KMeans Clustering in PCA space (2D)')
        plt.legend()
        os.makedirs(os.path.join("Results", "Clustering"), exist_ok=True)
        plt.savefig(os.path.join("Results", "Clustering", "PCA_ext2.png"))
        
        plt.close()

    def compute_size(self, amp_shift):
        _, h, w = amp_shift.shape
        b = (np.floor(np.amin((h, w)) * self.L)).astype(int) 
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)
        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        return h1, h2, w1, w2
    
    def apply_style(self, image):
        return self._apply_style(image)

    def _apply_style(self, img):

        if self.n_images_per_style == 0:
            return img
        
        if self.mode == "noise":
            n = random.randint(0, len(self.styles) - 1)
            style = self.styles[n]
            noise_magnitude = np.mean(style) * 0.01
            noise = np.random.normal(0, noise_magnitude, size=style.shape)
            style = style + noise

        elif self.mode == "interpolation":
            if random.random() < self.p_interpolation:
                n1, n2 = np.random.choice(len(self.styles), 2, replace=False)
                style1, style2 = self.styles[n1], self.styles[n2]
                style = 2*style2 - style1
            else:
                n = random.randint(0, len(self.styles) - 1)
                style = self.styles[n]
        
        W, H = img.size
        img_np = self.preprocess(img)

        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp, pha = np.abs(fft_np), np.angle(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        amp_shift[:, h1:h2, w1:w2] = style
        amp_ = np.fft.ifftshift(amp_shift, axes=(-2, -1))

        fft_ = amp_ * np.exp(1j * pha)
        img_np_ = np.fft.ifft2(fft_, axes=(-2, -1))
        img_np_ = np.real(img_np_)
        img_np__ = np.clip(np.round(img_np_), 0., 255.)

        img_with_style = self.deprocess(img_np__, (W, H))

        return img_with_style