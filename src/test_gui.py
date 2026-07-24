# Last updated: 2026-07-23

import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

from infer import load_model, predict


class InferApp:
    def __init__(self, root: tk.Tk):
        self.model = load_model()
        self.image_path = None

        tk.Button(root, text="이미지 선택", command=self.select_image).pack()
        tk.Button(root, text="추론 시작", command=self.run_inference).pack()
        self.result_label = tk.Label(root, text="결과 없음", justify="left", anchor="w")
        self.result_label.pack(fill="x")
        self.image_label = tk.Label(root)
        self.image_label.pack()

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.image_path = path

    def run_inference(self):
        if not self.image_path:
            return
        results = predict(self.model, self.image_path)

        boxes = results[0].boxes
        if len(boxes) == 0:
            self.result_label.config(text="탐지된 물체 없음")
        else:
            lines = []
            for box in boxes:
                cls_name = results[0].names[int(box.cls)]
                conf = float(box.conf)
                lines.append(f"{cls_name} ({conf:.2f})")
            self.result_label.config(text="\n".join(lines))

        annotated_bgr = results[0].plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]

        img = Image.fromarray(annotated_rgb)
        img.thumbnail((640, 640))
        photo = ImageTk.PhotoImage(img)

        self.image_label.configure(image=photo)
        self.image_label.image = photo


def main() -> None:
    root = tk.Tk()
    root.title("YOLO 추론 테스트")
    InferApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
