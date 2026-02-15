# ...existing code...
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from transformers import AutoModelForImageClassification

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader



class CheXpertDataset(Dataset):
    classes = ['Cardiomegaly', 'Edema', 'Consolidation', 'Pneumonia', 'No Finding']
    
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(os.path.expanduser(csv_path))
        df[self.classes] = df[self.classes].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
        df['Path'] = df['Path'].apply(lambda x: f"{'/'.join(csv_path.split('/')[:-2])}/{x}")
        df[self.classes] = df[self.classes].replace(-1, 0)  # Replace uncertain labels with 0
        self.dataframe = pd.concat([df['Path'], df[self.classes]], axis=1)
        
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        # Load the image
        image_path = row['Path']
        image = Image.open(image_path).convert('RGB')
        
        # Convert labels to a float tensor
        label = torch.tensor(row[1:].values.astype(float), dtype=torch.float32)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

def inference(model, images, rho, device):
    _, _, h, w = images.shape

    sub_images = nn.functional.interpolate(images, size=(int(rho * h), int(rho * w)), mode='bilinear', align_corners=False)
    sub_images = nn.functional.interpolate(sub_images, size=(h, w), mode='bilinear', align_corners=False)
    
    model.eval()
    with torch.no_grad():
        sub_images = sub_images.to(device)
        outputs = model(sub_images)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
    predictions = (torch.sigmoid(logits) > 0.5).float()
    confidences = torch.maximum(torch.sigmoid(logits), 1 - torch.sigmoid(logits)).detach()  # Confidence scores for predictions

    return logits, predictions, confidences


def evaluation(csv_path, model_name, batch_size=16, device=None):
    """
    data_root is the root directory for the image dataset, such as '/content/CheXpert-v1.0-small'
    csv_path is the path to the CSV file containing image metadata and labels, such as '/content/CheXpert-v1.0-small/valid.csv'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    dataset = CheXpertDataset(csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    # Load model and processor
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.to(device)

    # Define loss function
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Validation Phase
    model.eval()
    rho_list = [1 - i/16 for i in range(16)]
    # rho_list = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
    loss_list = {rho: [] for rho in rho_list}
    acc_list = {rho: [] for rho in rho_list}
    conf_list = {rho: [] for rho in rho_list}

    with torch.no_grad():
        for k, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            for rho in rho_list:
                logits, predictions, conf = inference(model, images, rho, device)
                loss_list[rho].append(criterion(logits, labels))
                acc_list[rho].append((predictions == labels).float())
                conf_list[rho].append(conf)

        loss_list = {rho: torch.cat(loss, dim=0).mean(dim=0) for rho, loss in loss_list.items()}
        acc_list = {rho: torch.cat(acc, dim=0).mean(dim=0) for rho, acc in acc_list.items()}
        conf_list = {rho: torch.cat(conf, dim=0) for rho, conf in conf_list.items()}
        
    for rho in rho_list:
        loss_dict = {c: f"{loss_list[rho][i].item():.4f}" for i, c in enumerate(dataset.classes)}
        print(f"[Rho: {rho}] Loss: {loss_dict}")

    for rho in rho_list:
        acc_dict = {c: f"{acc_list[rho][i].item():.4f}" for i, c in enumerate(dataset.classes)}
        print(f"[Rho: {rho}] Accuracy: {acc_dict}")

    # for rho in rho_list:
    #     ratio_list = (conf_list[rho_list[0]] > conf_list[rho]).float().mean(dim=0)
    #     ratio_dict = {c: f"{ratio_list[i].item():.4f}" for i, c in enumerate(dataset.classes)}
    #     print(f"Rate of confidence[{rho_list[0]}] > confidence[{rho}]: {ratio_dict}")

    for rho1 in rho_list:
        for rho2 in rho_list:
            if rho1 > rho2:
                rate_list = (conf_list[rho1] > conf_list[rho2]).float().mean(dim=0)
                rate_dict = {c: f"{rate_list[i].item():.4f}" for i, c in enumerate(dataset.classes)}
                print(f"Rate of confidence[{rho1}] > confidence[{rho2}]: {rate_dict}")


# If executed directly, example call
if __name__ == "__main__":
    CSV_PATH = "/home/cs/Documents/datasets/CheXpert-v1.0-small/valid.csv"
    DATA_ROOT = "/home/cs/Documents/datasets/CheXpert-v1.0-small"
    MODEL_NAME = "codewithdark/vit-chest-xray"
    evaluation(CSV_PATH, MODEL_NAME, batch_size=16)


    # [Rho: 1.0] Loss: {'Cardiomegaly': '0.1636', 'Edema': '0.1035', 'Consolidation': '0.0975', 'Pneumonia': '0.0454', 'No Finding': '0.1018'}
    # [Rho: 0.9375] Loss: {'Cardiomegaly': '0.2041', 'Edema': '0.1384', 'Consolidation': '0.1209', 'Pneumonia': '0.0507', 'No Finding': '0.1214'}
    # [Rho: 0.875] Loss: {'Cardiomegaly': '0.1894', 'Edema': '0.1257', 'Consolidation': '0.1141', 'Pneumonia': '0.0507', 'No Finding': '0.1111'}
    # [Rho: 0.8125] Loss: {'Cardiomegaly': '0.1976', 'Edema': '0.1307', 'Consolidation': '0.1170', 'Pneumonia': '0.0515', 'No Finding': '0.1135'}
    # [Rho: 0.75] Loss: {'Cardiomegaly': '0.2048', 'Edema': '0.1375', 'Consolidation': '0.1216', 'Pneumonia': '0.0520', 'No Finding': '0.1197'}
    # [Rho: 0.6875] Loss: {'Cardiomegaly': '0.2045', 'Edema': '0.1388', 'Consolidation': '0.1198', 'Pneumonia': '0.0519', 'No Finding': '0.1192'}
    # [Rho: 0.625] Loss: {'Cardiomegaly': '0.2224', 'Edema': '0.1487', 'Consolidation': '0.1278', 'Pneumonia': '0.0525', 'No Finding': '0.1280'}
    # [Rho: 0.5625] Loss: {'Cardiomegaly': '0.2393', 'Edema': '0.1591', 'Consolidation': '0.1385', 'Pneumonia': '0.0541', 'No Finding': '0.1394'}
    # [Rho: 0.5] Loss: {'Cardiomegaly': '0.2730', 'Edema': '0.1947', 'Consolidation': '0.1553', 'Pneumonia': '0.0595', 'No Finding': '0.1627'}
    # [Rho: 0.4375] Loss: {'Cardiomegaly': '0.2707', 'Edema': '0.1981', 'Consolidation': '0.1582', 'Pneumonia': '0.0605', 'No Finding': '0.1674'}
    # [Rho: 0.375] Loss: {'Cardiomegaly': '0.3081', 'Edema': '0.2375', 'Consolidation': '0.1778', 'Pneumonia': '0.0700', 'No Finding': '0.1968'}
    # [Rho: 0.3125] Loss: {'Cardiomegaly': '0.3478', 'Edema': '0.2740', 'Consolidation': '0.2131', 'Pneumonia': '0.0814', 'No Finding': '0.2583'}
    # [Rho: 0.25] Loss: {'Cardiomegaly': '0.4187', 'Edema': '0.3055', 'Consolidation': '0.2549', 'Pneumonia': '0.1033', 'No Finding': '0.3038'}
    # [Rho: 0.1875] Loss: {'Cardiomegaly': '0.5227', 'Edema': '0.3551', 'Consolidation': '0.3167', 'Pneumonia': '0.1288', 'No Finding': '0.3591'}
    # [Rho: 0.125] Loss: {'Cardiomegaly': '0.6107', 'Edema': '0.4528', 'Consolidation': '0.3769', 'Pneumonia': '0.1793', 'No Finding': '0.4853'}
    # [Rho: 0.0625] Loss: {'Cardiomegaly': '0.6930', 'Edema': '0.5262', 'Consolidation': '0.4553', 'Pneumonia': '0.3300', 'No Finding': '0.6674'}
    # [Rho: 1.0] Accuracy: {'Cardiomegaly': '0.9786', 'Edema': '0.9872', 'Consolidation': '0.9915', 'Pneumonia': '0.9872', 'No Finding': '0.9915'}
    # [Rho: 0.9375] Accuracy: {'Cardiomegaly': '0.9487', 'Edema': '0.9658', 'Consolidation': '0.9829', 'Pneumonia': '0.9829', 'No Finding': '0.9829'}
    # [Rho: 0.875] Accuracy: {'Cardiomegaly': '0.9701', 'Edema': '0.9658', 'Consolidation': '0.9915', 'Pneumonia': '0.9915', 'No Finding': '0.9829'}
    # [Rho: 0.8125] Accuracy: {'Cardiomegaly': '0.9573', 'Edema': '0.9701', 'Consolidation': '0.9915', 'Pneumonia': '0.9872', 'No Finding': '0.9829'}
    # [Rho: 0.75] Accuracy: {'Cardiomegaly': '0.9573', 'Edema': '0.9615', 'Consolidation': '0.9829', 'Pneumonia': '0.9872', 'No Finding': '0.9744'}
    # [Rho: 0.6875] Accuracy: {'Cardiomegaly': '0.9701', 'Edema': '0.9658', 'Consolidation': '0.9872', 'Pneumonia': '0.9829', 'No Finding': '0.9786'}
    # [Rho: 0.625] Accuracy: {'Cardiomegaly': '0.9274', 'Edema': '0.9573', 'Consolidation': '0.9786', 'Pneumonia': '0.9829', 'No Finding': '0.9744'}
    # [Rho: 0.5625] Accuracy: {'Cardiomegaly': '0.9188', 'Edema': '0.9573', 'Consolidation': '0.9786', 'Pneumonia': '0.9829', 'No Finding': '0.9573'}
    # [Rho: 0.5] Accuracy: {'Cardiomegaly': '0.8932', 'Edema': '0.9188', 'Consolidation': '0.9658', 'Pneumonia': '0.9829', 'No Finding': '0.9573'}
    # [Rho: 0.4375] Accuracy: {'Cardiomegaly': '0.9103', 'Edema': '0.9231', 'Consolidation': '0.9487', 'Pneumonia': '0.9829', 'No Finding': '0.9615'}
    # [Rho: 0.375] Accuracy: {'Cardiomegaly': '0.8974', 'Edema': '0.9060', 'Consolidation': '0.9231', 'Pneumonia': '0.9744', 'No Finding': '0.9786'}
    # [Rho: 0.3125] Accuracy: {'Cardiomegaly': '0.8803', 'Edema': '0.8632', 'Consolidation': '0.8932', 'Pneumonia': '0.9744', 'No Finding': '0.9487'}
    # [Rho: 0.25] Accuracy: {'Cardiomegaly': '0.8291', 'Edema': '0.8120', 'Consolidation': '0.8846', 'Pneumonia': '0.9701', 'No Finding': '0.9359'}
    # [Rho: 0.1875] Accuracy: {'Cardiomegaly': '0.7778', 'Edema': '0.8419', 'Consolidation': '0.8590', 'Pneumonia': '0.9701', 'No Finding': '0.8932'}
    # [Rho: 0.125] Accuracy: {'Cardiomegaly': '0.6966', 'Edema': '0.7991', 'Consolidation': '0.8590', 'Pneumonia': '0.9658', 'No Finding': '0.8462'}
    # [Rho: 0.0625] Accuracy: {'Cardiomegaly': '0.5598', 'Edema': '0.7650', 'Consolidation': '0.8547', 'Pneumonia': '0.9530', 'No Finding': '0.5726'}
    # Rate of confidence[1.0] > confidence[0.9375]: {'Cardiomegaly': '0.7949', 'Edema': '0.5726', 'Consolidation': '0.8376', 'Pneumonia': '0.6111', 'No Finding': '0.6197'}
    # Rate of confidence[1.0] > confidence[0.875]: {'Cardiomegaly': '0.7735', 'Edema': '0.5726', 'Consolidation': '0.8291', 'Pneumonia': '0.7650', 'No Finding': '0.4402'}
    # Rate of confidence[1.0] > confidence[0.8125]: {'Cardiomegaly': '0.7821', 'Edema': '0.5769', 'Consolidation': '0.8205', 'Pneumonia': '0.7009', 'No Finding': '0.4872'}
    # Rate of confidence[1.0] > confidence[0.75]: {'Cardiomegaly': '0.7778', 'Edema': '0.6026', 'Consolidation': '0.8333', 'Pneumonia': '0.6838', 'No Finding': '0.5470'}
    # Rate of confidence[1.0] > confidence[0.6875]: {'Cardiomegaly': '0.7906', 'Edema': '0.6068', 'Consolidation': '0.8162', 'Pneumonia': '0.6581', 'No Finding': '0.5385'}
    # Rate of confidence[1.0] > confidence[0.625]: {'Cardiomegaly': '0.8034', 'Edema': '0.6239', 'Consolidation': '0.8291', 'Pneumonia': '0.6368', 'No Finding': '0.6197'}
    # Rate of confidence[1.0] > confidence[0.5625]: {'Cardiomegaly': '0.8291', 'Edema': '0.6838', 'Consolidation': '0.8718', 'Pneumonia': '0.6325', 'No Finding': '0.6752'}
    # Rate of confidence[1.0] > confidence[0.5]: {'Cardiomegaly': '0.8590', 'Edema': '0.7222', 'Consolidation': '0.8932', 'Pneumonia': '0.6667', 'No Finding': '0.8333'}
    # Rate of confidence[1.0] > confidence[0.4375]: {'Cardiomegaly': '0.8504', 'Edema': '0.7137', 'Consolidation': '0.8675', 'Pneumonia': '0.6795', 'No Finding': '0.8462'}
    # Rate of confidence[1.0] > confidence[0.375]: {'Cardiomegaly': '0.8675', 'Edema': '0.7222', 'Consolidation': '0.8590', 'Pneumonia': '0.7863', 'No Finding': '0.9231'}
    # Rate of confidence[1.0] > confidence[0.3125]: {'Cardiomegaly': '0.8889', 'Edema': '0.8077', 'Consolidation': '0.8419', 'Pneumonia': '0.8419', 'No Finding': '0.9658'}
    # Rate of confidence[1.0] > confidence[0.25]: {'Cardiomegaly': '0.9060', 'Edema': '0.8419', 'Consolidation': '0.8632', 'Pneumonia': '0.8761', 'No Finding': '0.9786'}
    # Rate of confidence[1.0] > confidence[0.1875]: {'Cardiomegaly': '0.8932', 'Edema': '0.9274', 'Consolidation': '0.8932', 'Pneumonia': '0.9316', 'No Finding': '0.9658'}
    # Rate of confidence[1.0] > confidence[0.125]: {'Cardiomegaly': '0.9402', 'Edema': '0.9444', 'Consolidation': '0.9274', 'Pneumonia': '0.9615', 'No Finding': '0.9829'}
    # Rate of confidence[1.0] > confidence[0.0625]: {'Cardiomegaly': '0.9658', 'Edema': '0.9487', 'Consolidation': '0.9231', 'Pneumonia': '0.9744', 'No Finding': '0.9957'}
    # Rate of confidence[0.9375] > confidence[0.875]: {'Cardiomegaly': '0.3162', 'Edema': '0.5128', 'Consolidation': '0.3120', 'Pneumonia': '0.7094', 'No Finding': '0.2009'}
    # Rate of confidence[0.9375] > confidence[0.8125]: {'Cardiomegaly': '0.4060', 'Edema': '0.5427', 'Consolidation': '0.3419', 'Pneumonia': '0.7051', 'No Finding': '0.1752'}
    # Rate of confidence[0.9375] > confidence[0.75]: {'Cardiomegaly': '0.5427', 'Edema': '0.6239', 'Consolidation': '0.5043', 'Pneumonia': '0.7265', 'No Finding': '0.2991'}
    # Rate of confidence[0.9375] > confidence[0.6875]: {'Cardiomegaly': '0.5128', 'Edema': '0.5940', 'Consolidation': '0.4744', 'Pneumonia': '0.6496', 'No Finding': '0.3376'}
    # Rate of confidence[0.9375] > confidence[0.625]: {'Cardiomegaly': '0.6496', 'Edema': '0.6538', 'Consolidation': '0.6368', 'Pneumonia': '0.5812', 'No Finding': '0.5641'}
    # Rate of confidence[0.9375] > confidence[0.5625]: {'Cardiomegaly': '0.7521', 'Edema': '0.6667', 'Consolidation': '0.7222', 'Pneumonia': '0.6197', 'No Finding': '0.6795'}
    # Rate of confidence[0.9375] > confidence[0.5]: {'Cardiomegaly': '0.7778', 'Edema': '0.6923', 'Consolidation': '0.8291', 'Pneumonia': '0.6239', 'No Finding': '0.8932'}
    # Rate of confidence[0.9375] > confidence[0.4375]: {'Cardiomegaly': '0.7821', 'Edema': '0.7137', 'Consolidation': '0.7436', 'Pneumonia': '0.6581', 'No Finding': '0.8761'}
    # Rate of confidence[0.9375] > confidence[0.375]: {'Cardiomegaly': '0.8376', 'Edema': '0.7265', 'Consolidation': '0.7308', 'Pneumonia': '0.7991', 'No Finding': '0.9231'}
    # Rate of confidence[0.9375] > confidence[0.3125]: {'Cardiomegaly': '0.8632', 'Edema': '0.7863', 'Consolidation': '0.7393', 'Pneumonia': '0.8504', 'No Finding': '0.9530'}
    # Rate of confidence[0.9375] > confidence[0.25]: {'Cardiomegaly': '0.8932', 'Edema': '0.8504', 'Consolidation': '0.8376', 'Pneumonia': '0.8932', 'No Finding': '0.9615'}
    # Rate of confidence[0.9375] > confidence[0.1875]: {'Cardiomegaly': '0.8889', 'Edema': '0.9103', 'Consolidation': '0.8675', 'Pneumonia': '0.9274', 'No Finding': '0.9701'}
    # Rate of confidence[0.9375] > confidence[0.125]: {'Cardiomegaly': '0.9145', 'Edema': '0.9359', 'Consolidation': '0.8718', 'Pneumonia': '0.9573', 'No Finding': '0.9487'}
    # Rate of confidence[0.9375] > confidence[0.0625]: {'Cardiomegaly': '0.9145', 'Edema': '0.9573', 'Consolidation': '0.8718', 'Pneumonia': '0.9786', 'No Finding': '0.9744'}
    # Rate of confidence[0.875] > confidence[0.8125]: {'Cardiomegaly': '0.5983', 'Edema': '0.5385', 'Consolidation': '0.6026', 'Pneumonia': '0.4701', 'No Finding': '0.6111'}
    # Rate of confidence[0.875] > confidence[0.75]: {'Cardiomegaly': '0.7094', 'Edema': '0.5855', 'Consolidation': '0.7094', 'Pneumonia': '0.4316', 'No Finding': '0.7607'}
    # Rate of confidence[0.875] > confidence[0.6875]: {'Cardiomegaly': '0.6667', 'Edema': '0.6410', 'Consolidation': '0.6838', 'Pneumonia': '0.3846', 'No Finding': '0.7094'}
    # Rate of confidence[0.875] > confidence[0.625]: {'Cardiomegaly': '0.7308', 'Edema': '0.6282', 'Consolidation': '0.7350', 'Pneumonia': '0.4274', 'No Finding': '0.7949'}
    # Rate of confidence[0.875] > confidence[0.5625]: {'Cardiomegaly': '0.7949', 'Edema': '0.6410', 'Consolidation': '0.7692', 'Pneumonia': '0.4274', 'No Finding': '0.8376'}
    # Rate of confidence[0.875] > confidence[0.5]: {'Cardiomegaly': '0.8333', 'Edema': '0.6709', 'Consolidation': '0.8333', 'Pneumonia': '0.5342', 'No Finding': '0.9231'}
    # Rate of confidence[0.875] > confidence[0.4375]: {'Cardiomegaly': '0.8205', 'Edema': '0.6838', 'Consolidation': '0.7692', 'Pneumonia': '0.5855', 'No Finding': '0.9231'}
    # Rate of confidence[0.875] > confidence[0.375]: {'Cardiomegaly': '0.8590', 'Edema': '0.7222', 'Consolidation': '0.7564', 'Pneumonia': '0.7521', 'No Finding': '0.9487'}
    # Rate of confidence[0.875] > confidence[0.3125]: {'Cardiomegaly': '0.8718', 'Edema': '0.7821', 'Consolidation': '0.7991', 'Pneumonia': '0.8120', 'No Finding': '0.9658'}
    # Rate of confidence[0.875] > confidence[0.25]: {'Cardiomegaly': '0.8974', 'Edema': '0.8205', 'Consolidation': '0.8462', 'Pneumonia': '0.8462', 'No Finding': '0.9701'}
    # Rate of confidence[0.875] > confidence[0.1875]: {'Cardiomegaly': '0.8718', 'Edema': '0.9103', 'Consolidation': '0.8675', 'Pneumonia': '0.9145', 'No Finding': '0.9744'}
    # Rate of confidence[0.875] > confidence[0.125]: {'Cardiomegaly': '0.9188', 'Edema': '0.9274', 'Consolidation': '0.8932', 'Pneumonia': '0.9487', 'No Finding': '0.9615'}
    # Rate of confidence[0.875] > confidence[0.0625]: {'Cardiomegaly': '0.9274', 'Edema': '0.9487', 'Consolidation': '0.8718', 'Pneumonia': '0.9786', 'No Finding': '0.9744'}
    # Rate of confidence[0.8125] > confidence[0.75]: {'Cardiomegaly': '0.6111', 'Edema': '0.5427', 'Consolidation': '0.6667', 'Pneumonia': '0.5171', 'No Finding': '0.6795'}
    # Rate of confidence[0.8125] > confidence[0.6875]: {'Cardiomegaly': '0.6068', 'Edema': '0.5983', 'Consolidation': '0.6239', 'Pneumonia': '0.4872', 'No Finding': '0.6581'}
    # Rate of confidence[0.8125] > confidence[0.625]: {'Cardiomegaly': '0.7265', 'Edema': '0.6239', 'Consolidation': '0.7436', 'Pneumonia': '0.4487', 'No Finding': '0.8205'}
    # Rate of confidence[0.8125] > confidence[0.5625]: {'Cardiomegaly': '0.7735', 'Edema': '0.6239', 'Consolidation': '0.7350', 'Pneumonia': '0.4744', 'No Finding': '0.8291'}
    # Rate of confidence[0.8125] > confidence[0.5]: {'Cardiomegaly': '0.8205', 'Edema': '0.6880', 'Consolidation': '0.8333', 'Pneumonia': '0.5385', 'No Finding': '0.9359'}
    # Rate of confidence[0.8125] > confidence[0.4375]: {'Cardiomegaly': '0.7991', 'Edema': '0.6966', 'Consolidation': '0.7521', 'Pneumonia': '0.5769', 'No Finding': '0.9145'}
    # Rate of confidence[0.8125] > confidence[0.375]: {'Cardiomegaly': '0.8248', 'Edema': '0.7179', 'Consolidation': '0.7436', 'Pneumonia': '0.7607', 'No Finding': '0.9402'}
    # Rate of confidence[0.8125] > confidence[0.3125]: {'Cardiomegaly': '0.8504', 'Edema': '0.7863', 'Consolidation': '0.7906', 'Pneumonia': '0.8120', 'No Finding': '0.9573'}
    # Rate of confidence[0.8125] > confidence[0.25]: {'Cardiomegaly': '0.8846', 'Edema': '0.8248', 'Consolidation': '0.8376', 'Pneumonia': '0.8547', 'No Finding': '0.9744'}
    # Rate of confidence[0.8125] > confidence[0.1875]: {'Cardiomegaly': '0.8718', 'Edema': '0.9145', 'Consolidation': '0.8761', 'Pneumonia': '0.9188', 'No Finding': '0.9701'}
    # Rate of confidence[0.8125] > confidence[0.125]: {'Cardiomegaly': '0.9231', 'Edema': '0.9444', 'Consolidation': '0.8761', 'Pneumonia': '0.9530', 'No Finding': '0.9573'}
    # Rate of confidence[0.8125] > confidence[0.0625]: {'Cardiomegaly': '0.9145', 'Edema': '0.9573', 'Consolidation': '0.8761', 'Pneumonia': '0.9829', 'No Finding': '0.9786'}
    # Rate of confidence[0.75] > confidence[0.6875]: {'Cardiomegaly': '0.5128', 'Edema': '0.5214', 'Consolidation': '0.4658', 'Pneumonia': '0.4786', 'No Finding': '0.5214'}
    # Rate of confidence[0.75] > confidence[0.625]: {'Cardiomegaly': '0.6624', 'Edema': '0.6026', 'Consolidation': '0.6068', 'Pneumonia': '0.4615', 'No Finding': '0.7308'}
    # Rate of confidence[0.75] > confidence[0.5625]: {'Cardiomegaly': '0.7564', 'Edema': '0.6496', 'Consolidation': '0.7051', 'Pneumonia': '0.4573', 'No Finding': '0.7991'}
    # Rate of confidence[0.75] > confidence[0.5]: {'Cardiomegaly': '0.8291', 'Edema': '0.6923', 'Consolidation': '0.8376', 'Pneumonia': '0.5385', 'No Finding': '0.9316'}
    # Rate of confidence[0.75] > confidence[0.4375]: {'Cardiomegaly': '0.7863', 'Edema': '0.6795', 'Consolidation': '0.7265', 'Pneumonia': '0.5983', 'No Finding': '0.8889'}
    # Rate of confidence[0.75] > confidence[0.375]: {'Cardiomegaly': '0.8248', 'Edema': '0.7009', 'Consolidation': '0.7564', 'Pneumonia': '0.7521', 'No Finding': '0.9316'}
    # Rate of confidence[0.75] > confidence[0.3125]: {'Cardiomegaly': '0.8504', 'Edema': '0.7650', 'Consolidation': '0.7692', 'Pneumonia': '0.8291', 'No Finding': '0.9487'}
    # Rate of confidence[0.75] > confidence[0.25]: {'Cardiomegaly': '0.8803', 'Edema': '0.8205', 'Consolidation': '0.8419', 'Pneumonia': '0.8547', 'No Finding': '0.9615'}
    # Rate of confidence[0.75] > confidence[0.1875]: {'Cardiomegaly': '0.8632', 'Edema': '0.9103', 'Consolidation': '0.8675', 'Pneumonia': '0.9188', 'No Finding': '0.9701'}
    # Rate of confidence[0.75] > confidence[0.125]: {'Cardiomegaly': '0.9060', 'Edema': '0.9274', 'Consolidation': '0.8675', 'Pneumonia': '0.9573', 'No Finding': '0.9444'}
    # Rate of confidence[0.75] > confidence[0.0625]: {'Cardiomegaly': '0.9188', 'Edema': '0.9402', 'Consolidation': '0.8632', 'Pneumonia': '0.9829', 'No Finding': '0.9786'}
    # Rate of confidence[0.6875] > confidence[0.625]: {'Cardiomegaly': '0.6581', 'Edema': '0.5983', 'Consolidation': '0.6538', 'Pneumonia': '0.4701', 'No Finding': '0.7009'}
    # Rate of confidence[0.6875] > confidence[0.5625]: {'Cardiomegaly': '0.7650', 'Edema': '0.6068', 'Consolidation': '0.7308', 'Pneumonia': '0.4915', 'No Finding': '0.8376'}
    # Rate of confidence[0.6875] > confidence[0.5]: {'Cardiomegaly': '0.8248', 'Edema': '0.6795', 'Consolidation': '0.8291', 'Pneumonia': '0.5556', 'No Finding': '0.9231'}
    # Rate of confidence[0.6875] > confidence[0.4375]: {'Cardiomegaly': '0.8077', 'Edema': '0.6880', 'Consolidation': '0.7393', 'Pneumonia': '0.5897', 'No Finding': '0.8932'}
    # Rate of confidence[0.6875] > confidence[0.375]: {'Cardiomegaly': '0.8504', 'Edema': '0.7051', 'Consolidation': '0.7308', 'Pneumonia': '0.7692', 'No Finding': '0.9359'}
    # Rate of confidence[0.6875] > confidence[0.3125]: {'Cardiomegaly': '0.8547', 'Edema': '0.7735', 'Consolidation': '0.7650', 'Pneumonia': '0.8291', 'No Finding': '0.9487'}
    # Rate of confidence[0.6875] > confidence[0.25]: {'Cardiomegaly': '0.8718', 'Edema': '0.8248', 'Consolidation': '0.8419', 'Pneumonia': '0.8632', 'No Finding': '0.9658'}
    # Rate of confidence[0.6875] > confidence[0.1875]: {'Cardiomegaly': '0.8632', 'Edema': '0.9188', 'Consolidation': '0.8846', 'Pneumonia': '0.9231', 'No Finding': '0.9658'}
    # Rate of confidence[0.6875] > confidence[0.125]: {'Cardiomegaly': '0.9017', 'Edema': '0.9402', 'Consolidation': '0.8803', 'Pneumonia': '0.9444', 'No Finding': '0.9573'}
    # Rate of confidence[0.6875] > confidence[0.0625]: {'Cardiomegaly': '0.9188', 'Edema': '0.9444', 'Consolidation': '0.8761', 'Pneumonia': '0.9786', 'No Finding': '0.9829'}
    # Rate of confidence[0.625] > confidence[0.5625]: {'Cardiomegaly': '0.6838', 'Edema': '0.5897', 'Consolidation': '0.7009', 'Pneumonia': '0.5043', 'No Finding': '0.6538'}
    # Rate of confidence[0.625] > confidence[0.5]: {'Cardiomegaly': '0.8034', 'Edema': '0.6923', 'Consolidation': '0.8333', 'Pneumonia': '0.5812', 'No Finding': '0.9103'}
    # Rate of confidence[0.625] > confidence[0.4375]: {'Cardiomegaly': '0.7692', 'Edema': '0.6667', 'Consolidation': '0.7436', 'Pneumonia': '0.6282', 'No Finding': '0.8632'}
    # Rate of confidence[0.625] > confidence[0.375]: {'Cardiomegaly': '0.7949', 'Edema': '0.6880', 'Consolidation': '0.7094', 'Pneumonia': '0.7821', 'No Finding': '0.9103'}
    # Rate of confidence[0.625] > confidence[0.3125]: {'Cardiomegaly': '0.8376', 'Edema': '0.7607', 'Consolidation': '0.7308', 'Pneumonia': '0.8462', 'No Finding': '0.9274'}
    # Rate of confidence[0.625] > confidence[0.25]: {'Cardiomegaly': '0.8590', 'Edema': '0.8333', 'Consolidation': '0.8162', 'Pneumonia': '0.8889', 'No Finding': '0.9530'}
    # Rate of confidence[0.625] > confidence[0.1875]: {'Cardiomegaly': '0.8504', 'Edema': '0.9060', 'Consolidation': '0.8547', 'Pneumonia': '0.9188', 'No Finding': '0.9658'}
    # Rate of confidence[0.625] > confidence[0.125]: {'Cardiomegaly': '0.8932', 'Edema': '0.9359', 'Consolidation': '0.8718', 'Pneumonia': '0.9615', 'No Finding': '0.9487'}
    # Rate of confidence[0.625] > confidence[0.0625]: {'Cardiomegaly': '0.9103', 'Edema': '0.9573', 'Consolidation': '0.8632', 'Pneumonia': '0.9786', 'No Finding': '0.9615'}
    # Rate of confidence[0.5625] > confidence[0.5]: {'Cardiomegaly': '0.7308', 'Edema': '0.6667', 'Consolidation': '0.7906', 'Pneumonia': '0.5983', 'No Finding': '0.8932'}
    # Rate of confidence[0.5625] > confidence[0.4375]: {'Cardiomegaly': '0.7051', 'Edema': '0.6239', 'Consolidation': '0.6410', 'Pneumonia': '0.6453', 'No Finding': '0.8462'}
    # Rate of confidence[0.5625] > confidence[0.375]: {'Cardiomegaly': '0.7863', 'Edema': '0.6667', 'Consolidation': '0.6624', 'Pneumonia': '0.7735', 'No Finding': '0.9359'}
    # Rate of confidence[0.5625] > confidence[0.3125]: {'Cardiomegaly': '0.8162', 'Edema': '0.7350', 'Consolidation': '0.6880', 'Pneumonia': '0.8675', 'No Finding': '0.9316'}
    # Rate of confidence[0.5625] > confidence[0.25]: {'Cardiomegaly': '0.8462', 'Edema': '0.8034', 'Consolidation': '0.7692', 'Pneumonia': '0.9145', 'No Finding': '0.9402'}
    # Rate of confidence[0.5625] > confidence[0.1875]: {'Cardiomegaly': '0.8419', 'Edema': '0.9103', 'Consolidation': '0.8547', 'Pneumonia': '0.9487', 'No Finding': '0.9530'}
    # Rate of confidence[0.5625] > confidence[0.125]: {'Cardiomegaly': '0.8761', 'Edema': '0.9274', 'Consolidation': '0.8632', 'Pneumonia': '0.9615', 'No Finding': '0.9487'}
    # Rate of confidence[0.5625] > confidence[0.0625]: {'Cardiomegaly': '0.8803', 'Edema': '0.9402', 'Consolidation': '0.8590', 'Pneumonia': '0.9786', 'No Finding': '0.9701'}
    # Rate of confidence[0.5] > confidence[0.4375]: {'Cardiomegaly': '0.5684', 'Edema': '0.5085', 'Consolidation': '0.4701', 'Pneumonia': '0.5983', 'No Finding': '0.6111'}
    # Rate of confidence[0.5] > confidence[0.375]: {'Cardiomegaly': '0.7436', 'Edema': '0.6239', 'Consolidation': '0.5299', 'Pneumonia': '0.8120', 'No Finding': '0.8504'}
    # Rate of confidence[0.5] > confidence[0.3125]: {'Cardiomegaly': '0.7821', 'Edema': '0.7222', 'Consolidation': '0.5983', 'Pneumonia': '0.8761', 'No Finding': '0.9188'}
    # Rate of confidence[0.5] > confidence[0.25]: {'Cardiomegaly': '0.8547', 'Edema': '0.7821', 'Consolidation': '0.7521', 'Pneumonia': '0.9145', 'No Finding': '0.9316'}
    # Rate of confidence[0.5] > confidence[0.1875]: {'Cardiomegaly': '0.8376', 'Edema': '0.8974', 'Consolidation': '0.8248', 'Pneumonia': '0.9573', 'No Finding': '0.9359'}
    # Rate of confidence[0.5] > confidence[0.125]: {'Cardiomegaly': '0.8846', 'Edema': '0.9060', 'Consolidation': '0.8205', 'Pneumonia': '0.9615', 'No Finding': '0.9274'}
    # Rate of confidence[0.5] > confidence[0.0625]: {'Cardiomegaly': '0.8846', 'Edema': '0.9103', 'Consolidation': '0.8291', 'Pneumonia': '0.9744', 'No Finding': '0.9658'}
    # Rate of confidence[0.4375] > confidence[0.375]: {'Cardiomegaly': '0.7350', 'Edema': '0.6197', 'Consolidation': '0.5855', 'Pneumonia': '0.8034', 'No Finding': '0.8803'}
    # Rate of confidence[0.4375] > confidence[0.3125]: {'Cardiomegaly': '0.8034', 'Edema': '0.7179', 'Consolidation': '0.6581', 'Pneumonia': '0.8675', 'No Finding': '0.9487'}
    # Rate of confidence[0.4375] > confidence[0.25]: {'Cardiomegaly': '0.8504', 'Edema': '0.7991', 'Consolidation': '0.7778', 'Pneumonia': '0.9103', 'No Finding': '0.9615'}
    # Rate of confidence[0.4375] > confidence[0.1875]: {'Cardiomegaly': '0.8462', 'Edema': '0.9017', 'Consolidation': '0.8419', 'Pneumonia': '0.9658', 'No Finding': '0.9444'}
    # Rate of confidence[0.4375] > confidence[0.125]: {'Cardiomegaly': '0.8974', 'Edema': '0.9103', 'Consolidation': '0.8462', 'Pneumonia': '0.9573', 'No Finding': '0.9274'}
    # Rate of confidence[0.4375] > confidence[0.0625]: {'Cardiomegaly': '0.8974', 'Edema': '0.9017', 'Consolidation': '0.8333', 'Pneumonia': '0.9786', 'No Finding': '0.9701'}
    # Rate of confidence[0.375] > confidence[0.3125]: {'Cardiomegaly': '0.7650', 'Edema': '0.7350', 'Consolidation': '0.6966', 'Pneumonia': '0.8120', 'No Finding': '0.8547'}
    # Rate of confidence[0.375] > confidence[0.25]: {'Cardiomegaly': '0.8248', 'Edema': '0.7991', 'Consolidation': '0.8077', 'Pneumonia': '0.8718', 'No Finding': '0.9188'}
    # Rate of confidence[0.375] > confidence[0.1875]: {'Cardiomegaly': '0.8590', 'Edema': '0.8932', 'Consolidation': '0.8291', 'Pneumonia': '0.9701', 'No Finding': '0.9017'}
    # Rate of confidence[0.375] > confidence[0.125]: {'Cardiomegaly': '0.8846', 'Edema': '0.9103', 'Consolidation': '0.8547', 'Pneumonia': '0.9487', 'No Finding': '0.8889'}
    # Rate of confidence[0.375] > confidence[0.0625]: {'Cardiomegaly': '0.8974', 'Edema': '0.9103', 'Consolidation': '0.8462', 'Pneumonia': '0.9744', 'No Finding': '0.9402'}
    # Rate of confidence[0.3125] > confidence[0.25]: {'Cardiomegaly': '0.7607', 'Edema': '0.6923', 'Consolidation': '0.7650', 'Pneumonia': '0.8077', 'No Finding': '0.7735'}
    # Rate of confidence[0.3125] > confidence[0.1875]: {'Cardiomegaly': '0.8205', 'Edema': '0.8419', 'Consolidation': '0.8419', 'Pneumonia': '0.9359', 'No Finding': '0.7735'}
    # Rate of confidence[0.3125] > confidence[0.125]: {'Cardiomegaly': '0.8632', 'Edema': '0.8932', 'Consolidation': '0.8333', 'Pneumonia': '0.9615', 'No Finding': '0.8248'}
    # Rate of confidence[0.3125] > confidence[0.0625]: {'Cardiomegaly': '0.8803', 'Edema': '0.9103', 'Consolidation': '0.8248', 'Pneumonia': '0.9744', 'No Finding': '0.9060'}
    # Rate of confidence[0.25] > confidence[0.1875]: {'Cardiomegaly': '0.7650', 'Edema': '0.8718', 'Consolidation': '0.8205', 'Pneumonia': '0.8547', 'No Finding': '0.6966'}
    # Rate of confidence[0.25] > confidence[0.125]: {'Cardiomegaly': '0.7991', 'Edema': '0.8889', 'Consolidation': '0.8120', 'Pneumonia': '0.9658', 'No Finding': '0.8120'}
    # Rate of confidence[0.25] > confidence[0.0625]: {'Cardiomegaly': '0.7991', 'Edema': '0.8889', 'Consolidation': '0.8291', 'Pneumonia': '0.9829', 'No Finding': '0.8974'}
    # Rate of confidence[0.1875] > confidence[0.125]: {'Cardiomegaly': '0.7179', 'Edema': '0.8162', 'Consolidation': '0.6795', 'Pneumonia': '0.8632', 'No Finding': '0.7650'}
    # Rate of confidence[0.1875] > confidence[0.0625]: {'Cardiomegaly': '0.7479', 'Edema': '0.8547', 'Consolidation': '0.7692', 'Pneumonia': '0.9829', 'No Finding': '0.8932'}
    # Rate of confidence[0.125] > confidence[0.0625]: {'Cardiomegaly': '0.6111', 'Edema': '0.6624', 'Consolidation': '0.7393', 'Pneumonia': '0.9701', 'No Finding': '0.8034'}
