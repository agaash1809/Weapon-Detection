from ultralytics import YOLO
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
    
fromaddr = "agaash1809@gmail.com"
toaddr ="727722eucs009@skcet.ac.in"

model = YOLO("model/weapon_model.pt")  

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5  

source = r"input images\vijay1.jpg" 
cap = cv2.VideoCapture(source)

def mail(text):
    print(text)
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "WEAPON DETECTION SYSTEM"
    body = text
    msg.attach(MIMEText(body, 'plain'))
    filename = "mail/weapon.jpg"
    attachment = open("mail/weapon.jpg", "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload(attachment.read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', f"attachment; filename={filename}")
    msg.attach(p)
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(fromaddr, "xsxsrrixdcplmjvg")
    text = msg.as_string()
    s.sendmail(fromaddr, toaddr, text)
    s.quit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO prediction
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    # Draw detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  
            conf = float(box.conf[0])  
            x1, y1, x2, y2 = map(int, box.xyxy[0])  

            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imwrite(f'mail/weapon.jpg', frame)
            mail("WEAPON DETECTED \n Please Check It")

    cv2.imshow("Weapon Detection", frame)

    cv2.waitKey(0)
cv2.destroyAllWindows()
