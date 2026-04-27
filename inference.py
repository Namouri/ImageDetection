import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import smtplib
from email.mime.text import MIMEText

# ====================== INSTÄLLNINGAR ======================
MODEL_PATH = "best_resnet_finetune.pth"
IMG_SIZE = 224
CLASS_NAMES = ['avlossnade', 'defekta', 'intakta']
NUM_CAMERAS = 12  # Ändra till 9 eller 12

# ====================== LADDA MODELL ======================
def load_model():
    model = models.resnet50()
    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    )
    model.eval()
    return model

# ====================== FÖRBEHANDLA BILD ======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ====================== KLASSIFICERA EN BILD ======================
def classify_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(1)

    return {
        'class': CLASS_NAMES[predicted.item()],
        'confidence': round(confidence.item() * 100, 2)
    }

# ====================== KLASSIFICERA ALLA ARMAR ======================
GLOBAL_MODEL= load_model()
def classify_all_arms(image_folder):
    # Use global model to save time/memory
    results = {}

    for camera_id in range(1, NUM_CAMERAS + 1):
        arm_name = f"Arm {camera_id}"

        # Find latest image
        camera_images = sorted([
            f for f in os.listdir(image_folder)
            if f.startswith(f"camera{camera_id}_")
            and (f.endswith('.png') or f.endswith('.jpg'))
        ])

        if not camera_images:
            results[arm_name] = {'class': 'unknown', 'confidence': 0.0}
            continue

        latest_image = os.path.join(image_folder, camera_images[-1])
        result = classify_image(GLOBAL_MODEL, latest_image)

        # MANDATORY MAPPING FOR FRONTEND
        if result['class'] == 'intakta':
            result['class'] = 'ok'
        elif result['class'] == 'defekta':
            result['class'] = 'damaged'
        # 'avlossnade' remains as is

        results[arm_name] = result

    # Alert logic
    damaged_list = {k: v for k, v in results.items() if v['class'] in ['damaged', 'avlossnade']}
    if damaged_list:
        send_alert(damaged_list)

    return results

# ====================== SKICKA NOTIFIKATION ======================
def send_alert(damaged_arms):
    message_text = "Följande verktygstips behöver åtgärdas:\n\n"

    for arm, result in damaged_arms.items():
        message_text += f"• {arm}: {result['class']} "
        message_text += f"(Säkerhet: {result['confidence']}%)\n"

    message_text += "\nVänligen byt ut de skadade tipsen innan nästa körning."

    print("\n===== NOTIFIKATION =====")
    print(message_text)

    # Avkommentera detta när ni har email-uppgifterna från Ekobot
    # msg = MIMEText(message_text)
    # msg['Subject'] = '⚠️ Ekobot — Skadade verktygstips upptäckta'
    # msg['From'] = 'ekobot@gmail.com'
    # msg['To'] = 'lantbrukare@gmail.com'
    # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
    #     server.login('ekobot@gmail.com', 'ert-lösenord')
    #     server.send_message(msg)
    # print("Email skickat!")