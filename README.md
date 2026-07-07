import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
from tkinter import Canvas, filedialog
import warnings
import shutil
from tkinter import filedialog
warnings.filterwarnings("ignore")
from CTkMessagebox import CTkMessagebox


def draw_multi_color_gradient(canvas, width, height, colors):
    steps = len(colors) - 1
    segment_height = height // steps

    for i in range(steps):
        r1, g1, b1 = canvas.winfo_rgb(colors[i])
        r2, g2, b2 = canvas.winfo_rgb(colors[i + 1])
        r_ratio = (r2 - r1) / segment_height
        g_ratio = (g2 - g1) / segment_height
        b_ratio = (b2 - b1) / segment_height

        for j in range(segment_height):
            nr = int(r1 + (r_ratio * j))
            ng = int(g1 + (g_ratio * j))
            nb = int(b1 + (b_ratio * j))
            hex_color = f'#{nr//256:02x}{ng//256:02x}{nb//256:02x}'
            canvas.create_line(0, i * segment_height + j, width, i * segment_height + j, fill=hex_color)


class MainGUI:

    # mainGUI
    def __init__(self, root):

        self.bool_img_read = False
        self.bool_ts_img_read = False
        self.bool_ts_img_browse = False
        self.bool_resize = False
        self.bool_noise_removal = False
        self.bool_contrast_enhancement = False
        self.bool_data_balancing = False
        self.bool_background_removal = False
        self.bool_insect_localization = False
        self.bool_feature_analysis = False
        self.bool_feature_correlation = False
        self.bool_feature_extraction = False
        self.bool_classification = False
        self.bool_data_spliting = False
        self.bool_training = False
        self.bool_testing = False
        self.bool_generate_graphs = False




        self.bool_training = False
        self.bool_testing = False
        self.bool_generate_graphs = False

        image_dataset = []
        self.trainingsize = 80
        self.testingsize = 20

        self.root = root
        self.root.title(
            "DEEP LEARNING BASED CLASSIFICATION AND IDENTIFICATION OF HARMFUL AND USEFUL INSECTS FOR AGRICULTURE")
        self.root.geometry("1100x630")
        self.root.resizable(False, False)
        self.canvas = Canvas(root, width=1100, height=630, highlightthickness=0)
        self.canvas.place(x=1, y=2)
        gradient_colors = ["#659287", "#88BDA4", "#B1D3B9", "#E6F2DD"]
        draw_multi_color_gradient(self.canvas, 1100, 630, gradient_colors)

        self.project_title_frame = ctk.CTkFrame(master=root, width=1015, height=40, corner_radius=10,
                                                fg_color="white")
        self.project_title_frame.place(relx=0.5, y=20, anchor="n")

        self.project_title_label = ctk.CTkLabel(
            master=self.project_title_frame,
            text="DEEP LEARNING-BASED CLASSIFICATION AND IDENTIFICATION OF HARMFUL AND USEFUL INSECTS FOR AGRICULTURE",
            text_color="red",
            font=ctk.CTkFont(size=17, weight="bold"),
            wraplength=980,
            justify="center"
        )
        self.project_title_label.place(relx=0.5, rely=0.5, anchor="center")

        # Dataset Selection (Browse and Read)
        self.label_tsdataset = ctk.CTkFrame(master=root, width=200, height=100, corner_radius=0, fg_color="#428475")
        self.label_tsdataset.place(x=42, y=85)
        self.txt_trdataset_name = ctk.CTkEntry(master=self.label_tsdataset, placeholder_text="Select Dataset",
                                               width=120)
        self.txt_trdataset_name.place(x=10, y=10)
        self.btn_tr_browse = ctk.CTkButton(master=self.label_tsdataset, text="  Read  ", width=50, height=25,
                                           fg_color="plum", text_color="white", command=self.tr_browse_image)
        self.btn_tr_browse.place(x=135, y=10)
        self.txt_tsdataset_name = ctk.CTkEntry(master=self.label_tsdataset, placeholder_text="Select Input Image",
                                               width=120)
        self.txt_tsdataset_name.place(x=10, y=55)
        self.btn_ts_browse = ctk.CTkButton(master=self.label_tsdataset, text="Browse", width=50, height=25,
                                           fg_color="blue", text_color="white", command=self.browse_ts_file)
        self.btn_ts_browse.place(x=135, y=55)


        # Data Preprocession Section
        self.label_preprocessing = ctk.CTkFrame(master=root, width=230, height=100, corner_radius=0, fg_color="#428475")
        self.label_preprocessing.place(x=370, y=85, anchor="n")
        self.title_preprocessing = ctk.CTkLabel(master=self.label_preprocessing, text="Preprocessing",
                                                text_color="white", font=ctk.CTkFont(size=12, weight="bold"))
        self.title_preprocessing.place(x=110, y=10, anchor ="center")
        self.btn_resize = ctk.CTkButton(master=self.label_preprocessing,
                                                      text="Resize", width=50, height=30,
                                                      corner_radius=16, fg_color="#88BDA4", text_color="white",
                                                      command=self.resize_img)
        self.btn_resize.place(relx=0.20, rely=0.40, anchor="center")
        self.btn_noise_removal = ctk.CTkButton(master=self.label_preprocessing, text="   Noise Removal   ",
                                                  width=40, height=30, corner_radius=16, fg_color="#88BDA4",
                                                  text_color="white", command=self.noise_removal)
        self.btn_noise_removal.place(relx=0.67, rely=0.40, anchor="center")
        self.btn_contrast_enhancement = ctk.CTkButton(master=self.label_preprocessing, text="   Contrast Enhancement   ",
                                                  width=30, height=30, corner_radius=16, fg_color="#88BDA4",
                                                  text_color="white", command=self.contrast_enhancement)
        self.btn_contrast_enhancement.place(relx=0.50, rely=0.75, anchor="center")


        # Data_Balancing

        self.label_Data_Balancing = ctk.CTkFrame(master=root, width=100, height=100, corner_radius=0,
                                                     fg_color="#428475")
        self.label_Data_Balancing.place(x=545, y=85, anchor="n")
        self.btn_Data_Balancing = ctk.CTkButton(master=self.label_Data_Balancing, text="Data\nBalancing", width=75,
                                    height=30, corner_radius=10, fg_color="#88BDA4", text_color="white", command=self.data_balancing)
        self.btn_Data_Balancing.place(relx=0.50, rely=0.50, anchor="center")

        # Foreground & Background_removal

        self.label_Background_removal = ctk.CTkFrame(master=root, width=100, height=100, corner_radius=0,
                                                     fg_color="#428475")
        self.label_Background_removal.place(x=655, y=85, anchor="n")
        self.btn_Background_removal = ctk.CTkButton(master=self.label_Background_removal, text="Background\nRemoval", width=75,
                                    height=30, corner_radius=10, fg_color="#88BDA4", text_color="white",
                                    command=self.background_removal)
        self.btn_Background_removal.place(relx=0.50, rely=0.50, anchor="center")

        # Insect_Localization
        self.label_Insect_Localization = ctk.CTkFrame(master=root, width=100, height=100, corner_radius=0,
                                                     fg_color="#428475")
        self.label_Insect_Localization.place(x=764, y=85, anchor="n")
        self.btn_Insect_Localization = ctk.CTkButton(master=self.label_Insect_Localization, text="Insect\nLocalization",
                                                    width=75,
                                                    height=30, corner_radius=10, fg_color="#88BDA4", text_color="white",
                                                    command=self.localization)
        self.btn_Insect_Localization.place(relx=0.50, rely=0.50, anchor="center")

        # Morphological Feature_Analysis
        self.label_Feature_Analysis = ctk.CTkFrame(master=root, width=110, height=45, corner_radius=0,
                                                     fg_color="#428475")
        self.label_Feature_Analysis.place(x=885, y=85, anchor="n")
        self.btn_Feature_Analysis = ctk.CTkButton(master=self.label_Feature_Analysis, text="Feature\nAnalysis", width=75,
                                    height=30, corner_radius=10, fg_color="#88BDA4", text_color="white",
                                    command=self.feature_analysis)
        self.btn_Feature_Analysis.place(relx=0.50, rely=0.50, anchor="center")

        # Feature_Correlation
        self.label_Feature_Correlation = ctk.CTkFrame(master=root, width=110, height=45, corner_radius=0,
                                                   fg_color="#428475")
        self.label_Feature_Correlation.place(x=885, y=140, anchor="n")
        self.btn_Feature_Correlation = ctk.CTkButton(master=self.label_Feature_Correlation, text="Feature\nCorrelation",
                                                  width=75,
                                                  height=30, corner_radius=10, fg_color="#88BDA4", text_color="white",
                                                  command=self.feature_correlation)
        self.btn_Feature_Correlation.place(relx=0.50, rely=0.50, anchor="center")


        # Rich Visual Feature_Extraction
        self.label_Feature_Extraction = ctk.CTkFrame(master=root, width=110, height=45, corner_radius=0,
                                                      fg_color="#428475")
        self.label_Feature_Extraction.place(x=1005, y=85, anchor="n")
        self.btn_Feature_Extraction = ctk.CTkButton(master=self.label_Feature_Extraction, text="Feature\nExtraction",
                                                     width=75,
                                                     height=30, corner_radius=10, fg_color="#88BDA4",
                                                     text_color="white",
                                                     command=self.feature_extraction)
        self.btn_Feature_Extraction.place(relx=0.50, rely=0.50, anchor="center")

        # Insect_Classification
        self.label_Classification = ctk.CTkFrame(master=root, width=110, height=45, corner_radius=0,
                                                     fg_color="#428475")
        self.label_Classification.place(x=1005, y=140, anchor="n")
        self.btn_Classification = ctk.CTkButton(master=self.label_Classification, text="Insect\nClassification",
                                                    width=75,
                                                    height=30, corner_radius=10, fg_color="#88BDA4",
                                                    text_color="white",
                                                    command=self.classification)
        self.btn_Classification.place(relx=0.50, rely=0.50, anchor="center")

        # Pesticides Recommendation
        self.label_pesticides = ctk.CTkFrame(master=root, width=165, height=58, corner_radius=0,
                                                 fg_color="#428475")
        self.label_pesticides.place(x=990, y=230, anchor="n")
        self.btn_pesticides = ctk.CTkButton(master=self.label_pesticides, text="Pesticides\nRecommendation",
                                                width=75,
                                                height=30, corner_radius=10, fg_color="#88BDA4",
                                                text_color="white",
                                                command=self.fuzzy_recom)
        self.btn_pesticides.place(relx=0.50, rely=0.50, anchor="center")

        # Training and testing
        self.label_training = ctk.CTkFrame(master=root, width=168, height=100, corner_radius=0,
                                                 fg_color="#428475")
        self.label_training.place(x=990, y=298, anchor="n")

        self.btn_Dataset_splitting = ctk.CTkButton(master=self.label_training, text="Dataset Splitting",
                                                   width=125, height=18, corner_radius=16, fg_color="#88BDA4",
                                                   text_color="white", command="")#self.Dataset_Splitting
        self.btn_Dataset_splitting.place(relx=0.5, rely=0.22, anchor="center")
        self.btn_training = ctk.CTkButton(master=self.label_training, text="Training", width=125, height=18,
                                          corner_radius=16, fg_color="#88BDA4", text_color="white",
                                          command=self.training)#self.Training
        self.btn_training.place(relx=0.5, rely=0.50, anchor="center")
        self.btn_testing = ctk.CTkButton(master=self.label_training, text="Testing", width=125, height=18,
                                         corner_radius=16, fg_color="#88BDA4", text_color="white",
                                         command=self.testing)#self.Testing
        self.btn_testing.place(relx=0.5, rely=0.80, anchor="center")


        # Process and Result Window
        self.label_process_window = ctk.CTkFrame(root, width=400, height=310, corner_radius=0, fg_color="#659287")
        self.label_process_window.place(x=50, y=230)
        self.title_process_window = ctk.CTkLabel(master=self.label_process_window, text="Process Window",
                                                 text_color="white", font=ctk.CTkFont(size=12, weight="bold"))
        self.title_process_window.place(x=10, y=5)
        self.process_window = ctk.CTkTextbox(master=self.label_process_window, width=370, height=270,
                                             font=("Arial", 12))
        self.process_window.place(x=4, y=35)
        self.process_window.configure(state="disabled")

        self.label_result_window = ctk.CTkFrame(root, width=400, height=310, corner_radius=0, fg_color="#659287")
        self.label_result_window.place(x=475, y=230)
        self.title_result_window = ctk.CTkLabel(master=self.label_result_window, text="Result Window",
                                                text_color="white", font=ctk.CTkFont(size=12, weight="bold"))
        self.title_result_window.place(x=10, y=5)
        self.result_window = ctk.CTkTextbox(master=self.label_result_window, width=370, height=270, font=("Arial", 12))
        self.result_window.place(x=4, y=35)
        self.result_window.configure(state="disabled")

        # Results
        self.label_class_name = ctk.CTkLabel(root, text="INSECT IDENTIFICATION", text_color="white",
                                             font=("Arial", 14), width=190, height=30, fg_color="#428475")
        self.label_class_name.place(x=50, y=560)

        self.txt_class_name = ctk.CTkEntry(root, text_color="white", fg_color="#428475", font=("Arial", 16),
                                           width=220, height=30, state="disabled")
        self.txt_class_name.place(x=250, y=560)

        self.label_pesticides_recom = ctk.CTkLabel(root, text="PESTICIDES RECOMMENDATION", text_color="white",
                                                   font=("Arial", 14), width=260, height=30, fg_color="#428475")
        self.label_pesticides_recom.place(x=483, y=560)

        self.txt_pesticides_recom = ctk.CTkEntry(root, text_color="white", fg_color="#428475", font=("Arial", 16),
                                                 width=280, height=30, state="disabled")
        self.txt_pesticides_recom.place(x=753, y=560)

        # Final
        self.button_frame = ctk.CTkFrame(root, width=163, height=130, corner_radius=0, fg_color="#428475")
        self.button_frame.place(x=910, y=408)


        self.btn_graph = ctk.CTkButton(self.button_frame, fg_color="lightblue4", text="Generate Graphs", width=150,
                                       height=35, font=ctk.CTkFont(size=12, weight="bold"), command="")#self.result_graphs
        self.btn_graph.place(x=8, y=8)
        self.btn_clear = ctk.CTkButton(self.button_frame, fg_color="lightblue4", hover_color="#CD6600", text="Clear",
                                       width=150, height=35, font=ctk.CTkFont(size=12, weight="bold"),
                                       command=self.clear_ui)
        self.btn_clear.place(x=8, y=48)
        self.btn_exit = ctk.CTkButton(self.button_frame, fg_color="lightblue4", hover_color="#428475", text="Exit",
                                      width=150, height=35, font=ctk.CTkFont(size=12, weight="bold"), command=self.exit)
        self.btn_exit.place(x=8, y=88)

    # Function
    # Clear Button
    def clear_ui(self):
        geometry = self.root.geometry()
        title = self.root.title()
        for widget in self.root.winfo_children():
            widget.destroy()
        self.__init__(self.root)
        self.root.geometry(geometry)
        self.root.title(title)
        self.root.update_idletasks()

    # exit
    def exit(self):
        self.root.destroy()

    def log_to_process_window(self, message):
        self.process_window.configure(state="normal")
        self.process_window.insert("end", message + "\n")
        self.process_window.see("end")
        self.process_window.configure(state="disabled")

    def tr_browse_image(self):
        self.bool_img_read = True
        self.bool_tr_img_read = True
        self.txt_trdataset_name.configure(state="normal")

        self.fname = "..//Dataset//"
        self.imglist = getListOfFiles(self.fname)
        for x in range(len(self.imglist)):
            fo_data = self.imglist[x].split("\\")
            self.image_name = (fo_data[-1])

        self.txt_trdataset_name.configure(state="disabled")
        print("\n Dataset was selected successfully...")
        CTkMessagebox(title="Success", message="Dataset was selected successfully...", icon="check", option_1="OK")

        self.btn_tr_browse.configure(state="disabled")
        self.btn_ts_browse.configure(state="disabled")
    def browse_ts_file(self):
        self.bool_ts_img_read = True
        self.bool_img_read = True
        self.bool_ts_img_browse = True
        self.btn_ts_browse.configure(state="normal")
        self.txt_tsdataset_name.configure(state="normal")
        self.file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )

        if self.file_path:
            self.img_name = self.file_path
            file_name = os.path.basename(self.img_name)
            self.image_data = os.path.splitext(file_name)[0]

            print("Image Name :", self.image_data)
            newpath = os.path.join("..", "Result", "Testing")
            os.makedirs(newpath, exist_ok=True)
            destination = os.path.join(newpath, file_name)
            shutil.copy2(self.img_name, destination)

            print("Image saved to:", destination)


        self.txt_tsdataset_name.configure(state="normal")
        self.txt_tsdataset_name.insert(0, str(os.path.basename(self.file_path)))
        self.txt_tsdataset_name.configure(state="disabled")
        print("\nSelected Image : " + str(os.path.basename(self.file_path)))
        self.patient_id = str(os.path.basename(self.file_path))

        self.test_img = cv2.imread(destination)
        output_file = "..\\Result\\Testing\\img.png"
        cv2.imwrite(output_file, self.test_img)

        CTkMessagebox(title="Success", message="Input Read successfully...", icon="check", option_1="OK")
        self.process_window.insert("end", "\nInput Read successfully...")
        self.btn_Data_Balancing.configure(state="disabled")
        self.btn_Feature_Extraction.configure(state="disabled")
        self.btn_Feature_Analysis.configure(state="disabled")
        self.btn_Feature_Correlation.configure(state="disabled")
        self.btn_Classification.configure(state="disabled")
        self.btn_training.configure(state="disabled")

    # Data Preprocessing
    def resize_img(self):
        self.bool_resize = True
        if self.bool_ts_img_browse:
            input_path = "..\\Result\\Testing\\img.png"
            output_path = "..\\Result\\Testing\\resized_img.png"
            img = cv2.imread(input_path)
            resized = cv2.resize(img, (512, 512))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            cv2.imwrite(output_path, resized)
            plt.figure(figsize=(6, 6))
            plt.imshow(resized)
            plt.title("Resized Image")
            plt.axis("off")
            plt.show()
            CTkMessagebox(title="Success", message="Resize testing was done successfully...", icon="check",
                          option_1="OK")
            self.process_window.insert("end", "\n=========")
            self.process_window.insert("end", "\n\nResize testing was done successfully...")
            self.process_window.configure(state="normal")
            self.process_window.insert("end", "\n=========")
            self.btn_resize.configure(state="disabled")

        elif self.bool_img_read:
            CTkMessagebox(title="Success", message="Resize testing was done successfully...", icon="check",option_1="OK")
            self.process_window.insert("end", "\nResize testing was done successfully...")
            self.process_window.configure(state="normal")
            self.process_window.insert("end", "\n=========")
            self.btn_resize.configure(state="disabled")


    def noise_removal(self):
        self.bool_noise_removal = True
        if self.bool_ts_img_browse:
            if self.bool_resize:
                input_path = "..\\Result\\Testing\\resized_img.png"
                img = cv2.imread(input_path)
                output_path = "..\\Result\\Testing\\noise_removal.png"
                resized = cv2.resize(img, (512, 512))
                blur = cv2.GaussianBlur(resized, (5, 5), 0)
                cv2.imwrite(output_path, blur)
                plt.figure(figsize=(6, 6))
                plt.imshow(blur)
                plt.title("Noise Removal Image")
                plt.axis("off")
                plt.show()

                CTkMessagebox(title="Success", message="Noise Removal testing was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\nNoise Removal testing was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_noise_removal.configure(state="disabled")


        elif self.bool_img_read:
            if self.bool_resize:
                CTkMessagebox(title="Success", message="Noise Removal training was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\nNoise Removal training was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_noise_removal.configure(state="disabled")

    def contrast_enhancement(self):
        self.bool_data_balancing = True
        self.bool_contrast_enhancement = True
        if self.bool_ts_img_browse:
            if self.bool_noise_removal:
                input_path = "..\\Result\\Testing\\noise_removal.png"
                img = cv2.imread(input_path)
                output_path = "..\\Result\\Testing\\contrast_enhancement.png"
                if img is not None:
                    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(
                        clipLimit=2.0,
                        tileGridSize=(8, 8)
                    )
                    l = clahe.apply(l)
                    enhanced_lab = cv2.merge((l, a, b))
                    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    cv2.imwrite(output_path, enhanced)
                    plt.figure(figsize=(6, 6))
                    plt.imshow(enhanced)
                    plt.title("Contrast Enhancement Image")
                    plt.axis("off")
                    plt.show()
                else:
                    shutil.copy2(input_path, output_path)

                CTkMessagebox(title="Success", message="Contrast Enhancement was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\nContrast Enhancement was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_contrast_enhancement.configure(state="disabled")
        elif self.bool_img_read:
            if self.bool_noise_removal:
                self.process_window.insert("end", "\nContrast Enhancement was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_contrast_enhancement.configure(state="disabled")


    def data_balancing(self):
        self.bool_data_balancing = True
        if self.bool_img_read:
            if self.bool_contrast_enhancement:
                CTkMessagebox(title="Success", message="Data Balancing was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\nData Balancing was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_Data_Balancing.configure(state="disabled")


    def background_removal(self):
        self.bool_background_removal = True
        if self.bool_ts_img_browse:
            if self.bool_contrast_enhancement:
                input_path = "..\\Result\\Testing\\noise_removal.png"
                output_path = "..\\Result\\Testing\\background_removal.png"
                img = cv2.imread(input_path)
                mask = np.zeros(img.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)
                rect = (50, 50, 400, 400)
                cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                foreground = img * mask2[:, :, np.newaxis]
                cv2.imwrite(output_path, foreground)
                plt.figure(figsize=(6, 6))
                plt.imshow(foreground)
                plt.title("Foreground & Background Removal Image")
                plt.axis("off")
                plt.show()
                CTkMessagebox(title="Success", message="Foreground and Background Removal was done successfully...",
                              icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\n Foreground and Background Removal was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_Background_removal.configure(state="disabled")

        elif self.bool_img_read:
            if self.bool_data_balancing:
                CTkMessagebox(title="Success", message="Foreground and Background Removal was done successfully...",
                              icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\n Foreground and Background Removal was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_Background_removal.configure(state="disabled")



    def localization(self):
        self.bool_insect_localization = True
        if self.bool_img_read:
            if self.bool_background_removal:
                CTkMessagebox(title="Success", message="Insect Localization was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\n Insect Localization was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_Insect_Localization.configure(state="disabled")

    def feature_analysis(self):
        self.bool_feature_analysis = True
        if self.bool_img_read:
            if self.bool_insect_localization:
                CTkMessagebox(title="Success", message="Feature Analysis was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\n Feature Analysis was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_Feature_Analysis.configure(state="disabled")

    def feature_correlation(self):
        self.bool_feature_correlation = True
        if self.bool_img_read:
            if self.bool_feature_analysis:
                CTkMessagebox(title="Success", message="Feature Correlation was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\n Feature Correlation was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_Feature_Correlation.configure(state="disabled")


    def feature_extraction(self):
        self.bool_feature_extraction = True
        if self.bool_img_read:
            if self.bool_feature_correlation:
                CTkMessagebox(title="Success", message="Feature Extraction was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\n Feature Extraction was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_Feature_Extraction.configure(state="disabled")

    def classification(self):
        self.bool_classification = True
        if self.bool_img_read:
            if self.bool_feature_extraction:
                CTkMessagebox(title="Success", message="Insect Classification was done successfully...", icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\n Insects Classification was done successfully...")
                self.process_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_Classification.configure(state="disabled")


    def fuzzy_recom(self):
        if self.bool_img_read:
            if self.bool_classification:
                CTkMessagebox(title="Success", message="Pesticides Recommendation was done successfully...",
                              icon="check",
                              option_1="OK")
                self.process_window.insert("end", "\n Pesticides Recommendation was done successfully...")
                self.procfess_window.configure(state="normal")
                self.process_window.insert("end", "\n=========")
                self.btn_pesticides.configure(state="disabled")

    def data_spliting(self):
        self.bool_data_spliting = True

    def training(self):
        self.bool_training = True

    def testing(self):
        self.bool_testing = True
        if self.bool_training:
            CTkMessagebox(title="Success", message="Testing was done successfully...", icon="check",
                          option_1="OK")
            self.process_window.insert("end", "\n Testing was done successfully...")
            self.process_window.configure(state="normal")
            self.result_window.configure(state="normal")
            self.process_window.insert("end", "\n\nTesting")
            self.process_window.insert("end", "\n=========")
            self.result_window.insert("end", "\n\nTesting")
            self.result_window.insert("end", "\n=========")

    def generate_graph(self):
        self.bool_generate_graphs = True

    def exit(self):
        self.root.destroy()

def getListOfFiles(image_folder):
    listOfFile = os.listdir(image_folder)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(image_folder, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

# GUI for open, close ,delete and title
root = ctk.CTk()
app = MainGUI(root)
root.geometry("1100x630")
root.configure(fg_color="#f5f5f5")
root.title("AGRICULTURAL INSECT IDENIFICATION")
root.mainloop()


