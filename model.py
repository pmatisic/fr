import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import json
import os
import time
from dataset import ImageProcessor, DataExtractor
from sklearn.model_selection import train_test_split
import numpy as np

class AgeGenderDataset(Dataset):
    def __init__(self, images_list, wiki_dir, imdb_dir, transform=None):
        print("Entered AgeGenderDataset.__init__")
        self.images_list = images_list
        self.wiki_dir = wiki_dir
        self.imdb_dir = imdb_dir
        self.transform = transform

    def __len__(self):
        print("Entered AgeGenderDataset.__len__")
        return len(self.images_list)

    def __getitem__(self, idx):
        print(f"Entered AgeGenderDataset.__getitem__ with idx={idx}")
        img_path, age, gender = self.images_list[idx]
        full_img_path = None
        if os.path.exists(os.path.join(self.wiki_dir, img_path)):
            full_img_path = os.path.join(self.wiki_dir, img_path)
        elif os.path.exists(os.path.join(self.imdb_dir, img_path)):
            full_img_path = os.path.join(self.imdb_dir, img_path)
        
        if full_img_path and os.path.exists(full_img_path):
            image = Image.open(full_img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long)
        else:
            return None, None, None

class AgeGenderModel(torch.nn.Module):
    def __init__(self):
        print("Entered AgeGenderModel.__init__")
        super(AgeGenderModel, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier[1] = torch.nn.Linear(self.efficientnet.classifier[1].in_features, 1024)
        self.fc_age = torch.nn.Linear(1024, 1)
        self.fc_gender = torch.nn.Linear(1024, 2)

    def forward(self, x):
        print("Entered AgeGenderModel.forward")
        features = self.efficientnet(x)
        age = self.fc_age(features)
        gender = self.fc_gender(features)
        return age, gender

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, num_epochs=10):
        print("Entered Trainer.__init__")
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.criterion_age = torch.nn.MSELoss()
        self.criterion_gender = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_log = []
        self.test_log = []
        self.batch_log = []
        self.stability_log = []
        self.robustness_log = []

    def train(self):
        print("Entered Trainer.train")
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch+1}/{self.num_epochs}")
            self.model.train()
            running_loss = 0.0
            epoch_start_time = time.time()
            for i, (images, ages, genders) in enumerate(self.train_loader):
                batch_start_time = time.time()
                print(f"Processing batch {i+1}/{len(self.train_loader)}")
                if images is None:
                    continue
                images, ages, genders = images.to(self.device), ages.to(self.device), genders.to(self.device)
                
                self.optimizer.zero_grad()
                outputs_age, outputs_gender = self.model(images)
                
                loss_age = self.criterion_age(outputs_age.squeeze(), ages)
                loss_gender = self.criterion_gender(outputs_gender, genders)
                loss = loss_age + loss_gender
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                print(f"Finished batch {i+1}/{len(self.train_loader)}, time taken: {batch_time:.2f}s")
                self.batch_log.append([epoch+1, i+1, loss.item(), batch_time])
            
            avg_train_loss = running_loss / len(self.train_loader)
            avg_test_loss = self.evaluate()

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Time taken: {epoch_time:.2f}s")

            self.train_log.append([epoch+1, avg_train_loss, epoch_time])
            self.test_log.append([epoch+1, avg_test_loss, epoch_time])
        
        torch.save(self.model.state_dict(), 'results/age_gender_model.pth')
        self.save_logs()
        self.test_stability()
        self.test_robustness()

    def evaluate(self):
        print("Entered Trainer.evaluate")
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, ages, genders in self.test_loader:
                if images is None:
                    continue
                images, ages, genders = images.to(self.device), ages.to(self.device), genders.to(self.device)
                outputs_age, outputs_gender = self.model(images)
                
                loss_age = self.criterion_age(outputs_age.squeeze(), ages)
                loss_gender = self.criterion_gender(outputs_gender, genders)
                loss = loss_age + loss_gender
                
                test_loss += loss.item()
        
        return test_loss / len(self.test_loader)

    def save_logs(self):
        train_df = pd.DataFrame(self.train_log, columns=["Epoch", "Train Loss", "Time"])
        test_df = pd.DataFrame(self.test_log, columns=["Epoch", "Test Loss", "Time"])
        batch_df = pd.DataFrame(self.batch_log, columns=["Epoch", "Batch", "Batch Loss", "Batch Time"])
        stability_df = pd.DataFrame(self.stability_log, columns=["Epoch", "Image Index", "Original Prediction", "Noisy Prediction", "Rotation Prediction", "Brightness Prediction", "Contrast Prediction"])
        robustness_df = pd.DataFrame(self.robustness_log, columns=["Epoch", "Image Index", "Original Prediction", "Adversarial Prediction"])

        train_df.to_csv('results/train_log.csv', index=False)
        test_df.to_csv('results/test_log.csv', index=False)
        batch_df.to_csv('results/batch_log.csv', index=False)
        stability_df.to_csv('results/stability_log.csv', index=False)
        robustness_df.to_csv('results/robustness_log.csv', index=False)

    def test_stability(self):
        print("Entered Trainer.test_stability")
        self.model.eval()
        with torch.no_grad():
            for i, (images, ages, genders) in enumerate(self.test_loader):
                if images is None:
                    continue
                images, ages, genders = images.to(self.device), ages.to(self.device), genders.to(self.device)
                original_predictions = self.model(images)
                
                # Add noise
                noisy_images = images + torch.randn_like(images) * 0.1
                noisy_predictions = self.model(noisy_images)
                
                # Rotate images
                rotated_images = torch.rot90(images, k=1, dims=[2, 3])
                rotated_predictions = self.model(rotated_images)
                
                # Adjust brightness
                brightness_images = torch.clamp(images * 1.5, 0, 1)
                brightness_predictions = self.model(brightness_images)
                
                # Adjust contrast
                contrast_images = torch.clamp((images - 0.5) * 1.5 + 0.5, 0, 1)
                contrast_predictions = self.model(contrast_images)
                
                for j in range(len(images)):
                    self.stability_log.append([i+1, j, original_predictions[j].cpu().numpy(), noisy_predictions[j].cpu().numpy(), rotated_predictions[j].cpu().numpy(), brightness_predictions[j].cpu().numpy(), contrast_predictions[j].cpu().numpy()])

    def test_robustness(self):
        print("Entered Trainer.test_robustness")
        self.model.eval()
        with torch.no_grad():
            for i, (images, ages, genders) in enumerate(self.test_loader):
                if images is None:
                    continue
                images, ages, genders = images.to(self.device), ages.to(self.device), genders.to(self.device)
                original_predictions = self.model(images)
                
                # Generate adversarial examples
                adversarial_images = self.generate_adversarial_examples(images, ages, genders)
                adversarial_predictions = self.model(adversarial_images)
                
                for j in range(len(images)):
                    self.robustness_log.append([i+1, j, original_predictions[j].cpu().numpy(), adversarial_predictions[j].cpu().numpy()])

    def generate_adversarial_examples(self, images, ages, genders, epsilon=0.1):
        print("Entered Trainer.generate_adversarial_examples")
        images.requires_grad = True
        outputs_age, outputs_gender = self.model(images)
        
        loss_age = self.criterion_age(outputs_age.squeeze(), ages)
        loss_gender = self.criterion_gender(outputs_gender, genders)
        loss = loss_age + loss_gender
        
        self.model.zero_grad()
        loss.backward()
        
        adversarial_images = images + epsilon * images.grad.sign()
        adversarial_images = torch.clamp(adversarial_images, 0, 1)
        
        return adversarial_images

class Evaluator:
    def __init__(self, model, data_loader, device):
        print("Entered Evaluator.__init__")
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def evaluate(self):
        print("Entered Evaluator.evaluate")
        self.model.to(self.device)
        self.model.eval()
        results = []
        with torch.no_grad():
            for images, ages, genders in self.data_loader:
                if images is None:
                    continue
                images, ages, genders = images.to(self.device), ages.to(self.device), genders.to(self.device)
                outputs_age, outputs_gender = self.model(images)
                
                predicted_ages = outputs_age.squeeze().cpu().numpy()
                predicted_genders = torch.argmax(outputs_gender, dim=1).cpu().numpy()
                
                for i in range(len(images)):
                    results.append([predicted_ages[i], predicted_genders[i], ages.cpu().numpy()[i], genders.cpu().numpy()[i]])

        results_df = pd.DataFrame(results, columns=["Predicted Age", "Predicted Gender", "Actual Age", "Actual Gender"])
        results_df.to_csv('results/predictions.csv', index=False)

def main():
    print("Entered main")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Use ImageProcessor to get valid images
    processor = ImageProcessor("data/wiki_crop/", "data/imdb_crop/")
    valid_images_json_path = "temp/valid_images.json"
    valid_images = processor.load_valid_images(valid_images_json_path)
    
    if not valid_images:
        valid_images, invalid_images_count = processor.get_valid_images(limit=100)
        processor.save_valid_images(valid_images, valid_images_json_path)
    else:
        print("Loaded valid images from JSON file.")
    
    train_data = [(item, 0, 0) for item in valid_images]  # Placeholder for age and gender
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    train_dataset = AgeGenderDataset(train_data, "data/wiki_crop/", "data/imdb_crop/", transform=train_transform)
    test_dataset = AgeGenderDataset(test_data, "data/wiki_crop/", "data/imdb_crop/", transform=train_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)  # Reduced batch size to 8
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    model = AgeGenderModel()
    trainer = Trainer(model, train_loader, test_loader, device)
    trainer.train()
    
    evaluator = Evaluator(model, test_loader, device)
    evaluator.evaluate()

if __name__ == '__main__':
    main()
