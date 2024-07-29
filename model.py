import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import os
import time
from dataset import ImageProcessor, DataExtractor
from sklearn.model_selection import train_test_split
import torchattacks

class AgeGenderDataset(Dataset):
    def __init__(self, images_list, wiki_dir, imdb_dir, transform=None):
        self.images_list = images_list
        self.wiki_dir = wiki_dir
        self.imdb_dir = imdb_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path, age, gender = self.images_list[idx]
        full_img_path = os.path.join(self.wiki_dir, img_path) if os.path.exists(os.path.join(self.wiki_dir, img_path)) else os.path.join(self.imdb_dir, img_path)
        
        if os.path.exists(full_img_path):
            image = Image.open(full_img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long)
        else:
            return None, None, None

class AgeGenderModel(torch.nn.Module):
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        self.efficientnet = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier[1] = torch.nn.Linear(self.efficientnet.classifier[1].in_features, 1024)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc_age = torch.nn.Linear(1024, 1)
        self.fc_gender = torch.nn.Linear(1024, 2)

    def forward(self, x):
        features = self.efficientnet(x)
        features = self.dropout(features)
        age = self.fc_age(features)
        gender = self.fc_gender(features)
        return age, gender

class GenderPredictionModel(torch.nn.Module):
    def __init__(self, base_model):
        super(GenderPredictionModel, self).__init__()
        self.base_model = base_model
    
    def forward(self, x):
        _, gender = self.base_model(x)
        return gender

class Trainer:
    def __init__(self, model, train_loader, test_loader, device, num_epochs=20, weight_decay=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.criterion_age = torch.nn.MSELoss()
        self.criterion_gender = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=weight_decay)
        self.train_log = []
        self.test_log = []
        self.batch_log = []
        self.stability_log = []
        self.robustness_log = []

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            print(f"Starting epoch {epoch+1}/{self.num_epochs}")
            self.model.train()
            running_loss = 0.0
            epoch_start_time = time.time()
            for i, (images, ages, genders) in enumerate(self.train_loader):
                batch_start_time = time.time()
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
                self.batch_log.append([epoch+1, i+1, loss.item(), batch_time])
                self.save_logs()

                print(f"Finished batch {i+1}/{len(self.train_loader)}, time taken: {batch_time:.2f}s")
                print(f"Batch {i+1} - Loss Age: {loss_age.item():.4f}, Loss Gender: {loss_gender.item():.4f}, Total Loss: {loss.item():.4f}")
                print(f"Predicted Age: {outputs_age.squeeze().detach().cpu().numpy()}")
                print(f"Predicted Gender: {torch.argmax(outputs_gender, dim=1).detach().cpu().numpy()}")
                print(f"Actual Age: {ages.detach().cpu().numpy()}")
                print(f"Actual Gender: {genders.detach().cpu().numpy()}")
                
                torch.cuda.empty_cache()
                
            avg_train_loss = running_loss / len(self.train_loader)
            avg_test_loss = self.evaluate()

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Time taken: {epoch_time:.2f}s")

            self.train_log.append([epoch+1, avg_train_loss, epoch_time])
            self.test_log.append([epoch+1, avg_test_loss, epoch_time])
            self.save_logs()
        
        torch.save(self.model.state_dict(), 'results/age_gender.pth')
        self.save_logs()
        self.test_stability()
        self.test_robustness()

    def evaluate(self):
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

                print(f"Predicted Age: {outputs_age.squeeze().cpu().numpy()}")
                print(f"Predicted Gender: {torch.argmax(outputs_gender, dim=1).cpu().numpy()}")
                print(f"Actual Age: {ages.cpu().numpy()}")
                print(f"Actual Gender: {genders.cpu().numpy()}")
        
        return test_loss / len(self.test_loader)

    def save_logs(self):
        train_df = pd.DataFrame(self.train_log, columns=["Epoch", "Train Loss", "Time"])
        test_df = pd.DataFrame(self.test_log, columns=["Epoch", "Test Loss", "Time"])
        batch_df = pd.DataFrame(self.batch_log, columns=["Epoch", "Batch", "Batch Loss", "Batch Time"])
        stability_df = pd.DataFrame(self.stability_log, columns=["Epoch", "Image Index", "Original Age Prediction", "Original Gender Prediction", "Noisy Age Prediction", "Noisy Gender Prediction", "Rotation Age Prediction", "Rotation Gender Prediction", "Brightness Age Prediction", "Brightness Gender Prediction", "Contrast Age Prediction", "Contrast Gender Prediction"])
        robustness_df = pd.DataFrame(self.robustness_log, columns=["Epoch", "Image Index", "Original Age Prediction", "Original Gender Prediction", "Adversarial Age Prediction", "Adversarial Gender Prediction"])

        train_df.to_csv('results/train.csv', index=False)
        test_df.to_csv('results/test.csv', index=False)
        batch_df.to_csv('results/batch.csv', index=False)
        stability_df.to_csv('results/stability.csv', index=False)
        robustness_df.to_csv('results/robustness.csv', index=False)

    def test_stability(self):
        self.model.eval()
        with torch.no_grad():
            for i, (images, ages, genders) in enumerate(self.test_loader):
                if images is None:
                    continue
                images, ages, genders = images.to(self.device), ages.to(self.device), genders.to(self.device)
                original_age_predictions, original_gender_predictions = self.model(images)
                
                # Add Gaussian noise
                noisy_images = images + torch.randn_like(images) * 0.1
                noisy_age_predictions, noisy_gender_predictions = self.model(noisy_images)
                
                # Rotate images
                rotated_images = torch.rot90(images, k=1, dims=[2, 3])
                rotated_age_predictions, rotated_gender_predictions = self.model(rotated_images)
                
                # Adjust brightness
                brightness_images = torch.clamp(images * 1.5, 0, 1)
                brightness_age_predictions, brightness_gender_predictions = self.model(brightness_images)
                
                # Adjust contrast
                contrast_images = torch.clamp((images - 0.5) * 1.5 + 0.5, 0, 1)
                contrast_age_predictions, contrast_gender_predictions = self.model(contrast_images)
                
                for j in range(len(images)):
                    self.stability_log.append([
                        i+1, j, 
                        original_age_predictions[j].cpu().numpy(), original_gender_predictions[j].cpu().numpy(),
                        noisy_age_predictions[j].cpu().numpy(), noisy_gender_predictions[j].cpu().numpy(),
                        rotated_age_predictions[j].cpu().numpy(), rotated_gender_predictions[j].cpu().numpy(),
                        brightness_age_predictions[j].cpu().numpy(), brightness_gender_predictions[j].cpu().numpy(),
                        contrast_age_predictions[j].cpu().numpy(), contrast_gender_predictions[j].cpu().numpy()
                    ])
                    self.save_logs()
                torch.cuda.empty_cache()

    def test_robustness(self):
        gender_model = GenderPredictionModel(self.model)
        gender_model.to(self.device)
        attack = torchattacks.PGD(gender_model, eps=0.1, alpha=0.01, steps=40)
        
        for i, (images, ages, genders) in enumerate(self.test_loader):
            if images is None:
                continue
            images, ages, genders = images.to(self.device), ages.to(self.device), genders.to(self.device)
            original_age_predictions, original_gender_predictions = self.model(images)
            
            adversarial_images = attack(images, genders)
            adversarial_age_predictions, adversarial_gender_predictions = self.model(adversarial_images)
            
            for j in range(len(images)):
                self.robustness_log.append([
                    i+1, j, 
                    original_age_predictions[j].detach().cpu().numpy(), original_gender_predictions[j].detach().cpu().numpy(),
                    adversarial_age_predictions[j].detach().cpu().numpy(), adversarial_gender_predictions[j].detach().cpu().numpy()
                ])
                self.save_logs()
            torch.cuda.empty_cache()

class Evaluator:
    def __init__(self, model, data_loader, device):
        self.model = model
        self.data_loader = data_loader
        self.device = device

    def evaluate(self):
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
                    print(f"Predicted Age: {predicted_ages[i]}, Predicted Gender: {predicted_genders[i]}, Actual Age: {ages.cpu().numpy()[i]}, Actual Gender: {genders.cpu().numpy()[i]}")

        results_df = pd.DataFrame(results, columns=["Predicted Age", "Predicted Gender", "Actual Age", "Actual Gender"])
        results_df.to_csv('results/predictions.csv', index=False)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    processor = ImageProcessor("data/wiki_crop/", "data/imdb_crop/")
    valid_images_json_path = "temp/valid_images.json"
    valid_images = processor.load_valid_images(valid_images_json_path)
    
    if not valid_images:
        valid_images, invalid_images_count = processor.get_valid_images(limit=999999)
        processor.save_valid_images(valid_images, valid_images_json_path)
    else:
        print("Loaded valid images from JSON file.")

    print(f"Number of valid images: {len(valid_images)}")
    print("Example of valid images data:", valid_images[:5])

    valid_images = valid_images[:10000]

    extractor = DataExtractor("data/wiki_crop/wiki.mat", "data/imdb_crop/imdb.mat")
    valid_images_set = set(valid_images)
    data = [(path, age, gender) for (path, age, gender) in extractor.all_data if path in valid_images_set]
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of testing samples: {len(test_data)}")
    
    train_dataset = AgeGenderDataset(train_data, "data/wiki_crop/", "data/imdb_crop/", transform=train_transform)
    test_dataset = AgeGenderDataset(test_data, "data/wiki_crop/", "data/imdb_crop/", transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=8)
    
    model = AgeGenderModel()
    trainer = Trainer(model, train_loader, test_loader, device)
    trainer.train()
    
    evaluator = Evaluator(model, test_loader, device)
    evaluator.evaluate()

if __name__ == '__main__':
    main()
