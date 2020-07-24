import cv2


def draw_rectangle(img, rect):
    (x1, y1, x2, y2) = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 0), 6)