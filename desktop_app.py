import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import database as db
from scanner import Scanner
import threading
import datetime
from tkinter import messagebox

# Set Algérie Télécom color palette
# Deep blue, crisp whites, cool grays. NO ORANGE.
ctk.set_appearance_mode("Light")  
ctk.set_default_color_theme("blue")  

# Custom Colors
BG_COLOR = "#F0F4F8"
SIDEBAR_COLOR = "#003366"
CARD_COLOR = "#FFFFFF"
TEXT_COLOR_DARK = "#001A33"
ACCENT_BLUE = "#004080"
SUCCESS_GREEN = "#198754"
DANGER_RED = "#DC3545"

class DesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Access Management System")
        self.geometry("1000x700")
        self.configure(fg_color=BG_COLOR)
        
        # Initialize Scanner in background
        self.scanner = None
        self.init_scanner_thread = threading.Thread(target=self.load_scanner)
        self.init_scanner_thread.start()
        
        # Grid Layout (1x2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # SIDEBAR
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color=SIDEBAR_COLOR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Access Manager", font=ctk.CTkFont(size=20, weight="bold"), text_color="#FFFFFF")
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 30))
        
        self.btn_dashboard = ctk.CTkButton(self.sidebar, text="Dashboard", fg_color="transparent", text_color="#FFFFFF", hover_color=ACCENT_BLUE, command=self.show_dashboard)
        self.btn_dashboard.grid(row=1, column=0, pady=10, padx=20, sticky="ew")
        
        self.btn_scan = ctk.CTkButton(self.sidebar, text="Scan ID", fg_color="transparent", text_color="#FFFFFF", hover_color=ACCENT_BLUE, command=self.show_scan)
        self.btn_scan.grid(row=2, column=0, pady=10, padx=20, sticky="ew")
        
        self.btn_profiles = ctk.CTkButton(self.sidebar, text="Profiles & Logs", fg_color="transparent", text_color="#FFFFFF", hover_color=ACCENT_BLUE, command=self.show_profiles)
        self.btn_profiles.grid(row=3, column=0, pady=10, padx=20, sticky="ew")
        
        # MAIN FRAMES
        self.dashboard_frame = ctk.CTkFrame(self, fg_color=BG_COLOR)
        self.scan_frame = ctk.CTkFrame(self, fg_color=BG_COLOR)
        self.profiles_frame = ctk.CTkFrame(self, fg_color=BG_COLOR)
        
        # SETUP FRAMES
        self.setup_dashboard()
        self.setup_scan()
        self.setup_profiles()
        
        # Show Dashboard initially
        self.show_dashboard()

    def load_scanner(self):
        self.scanner = Scanner()
        self.btn_capture.configure(state="normal", text="Capture & Extract")

    def show_dashboard(self):
        self.hide_all_frames()
        self.dashboard_frame.grid(row=0, column=1, sticky="nsew")
        self.update_dashboard()
        
    def show_scan(self):
        self.hide_all_frames()
        self.scan_frame.grid(row=0, column=1, sticky="nsew")
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.update_camera()

    def show_profiles(self):
        self.hide_all_frames()
        self.profiles_frame.grid(row=0, column=1, sticky="nsew")
        self.stop_camera()
        self.load_profiles()
        
    def hide_all_frames(self):
        self.dashboard_frame.grid_forget()
        self.scan_frame.grid_forget()
        self.profiles_frame.grid_forget()
        
    def stop_camera(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    # --- DASHBOARD ---
    def setup_dashboard(self):
        self.dashboard_frame.grid_columnconfigure(0, weight=1)
        self.dashboard_frame.grid_rowconfigure(0, weight=1)
        
        self.card = ctk.CTkFrame(self.dashboard_frame, fg_color=CARD_COLOR, corner_radius=15)
        self.card.grid(row=0, column=0, padx=50, pady=50, sticky="nsew")
        self.card.grid_columnconfigure(0, weight=1)
        self.card.grid_rowconfigure(1, weight=1)
        
        self.lbl_title = ctk.CTkLabel(self.card, text="Total Entries Today", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_COLOR_DARK)
        self.lbl_title.grid(row=0, column=0, pady=(50, 20))
        
        self.lbl_count = ctk.CTkLabel(self.card, text="0", font=ctk.CTkFont(size=72, weight="bold"), text_color=ACCENT_BLUE)
        self.lbl_count.grid(row=1, column=0, pady=20)
        
    def update_dashboard(self):
        count = db.get_total_entries_today()
        self.lbl_count.configure(text=str(count))

    # --- SCAN ID ---
    def setup_scan(self):
        self.scan_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(self.scan_frame, text="")
        self.video_label.grid(row=0, column=0, pady=20, padx=20)
        
        self.btn_capture = ctk.CTkButton(self.scan_frame, text="Loading Scanner...", font=ctk.CTkFont(size=18, weight="bold"), 
                                         fg_color=ACCENT_BLUE, hover_color="#002244", state="disabled",
                                         command=self.capture_and_extract)
        self.btn_capture.grid(row=1, column=0, pady=20, ipadx=20, ipady=10)
        
        self.current_frame = None

    def update_camera(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Convert BGR to RGB
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                # Resize for UI
                img = img.resize((640, 480))
                imgtk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                self.video_label.configure(image=imgtk)
            self.video_label.after(30, self.update_camera)

    def capture_and_extract(self):
        if self.current_frame is None or self.scanner is None:
            return
            
        frame_to_process = self.current_frame.copy()
        
        # Show loading state
        self.btn_capture.configure(state="disabled", text="Extracting...")
        self.update()
        
        # Run OCR in background to not freeze GUI
        threading.Thread(target=self._process_scan, args=(frame_to_process,)).start()

    def _process_scan(self, frame):
        result = self.scanner.scan_id(frame)
        self.btn_capture.configure(state="normal", text="Capture & Extract")
        
        if result is None or not result.get("nin"):
            messagebox.showerror("Error", "Could not extract NIN. Please align the card clearly.")
            return
            
        nin = result["nin"]
        user = db.get_user(nin)
        
        if user:
            # Workflow B: Existing
            self.show_checkin_popup(user)
        else:
            # Workflow A: New
            self.show_create_profile_popup(result)

    # --- WORKFLOW A: CREATE PROFILE ---
    def show_create_profile_popup(self, extracted_data):
        popup = ctk.CTkToplevel(self)
        popup.title("Create Profile")
        popup.geometry("400x500")
        popup.grab_set()
        
        popup.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(popup, text="NIN:").grid(row=0, column=0, padx=20, pady=10, sticky="w")
        ent_nin = ctk.CTkEntry(popup)
        ent_nin.insert(0, extracted_data.get("nin", ""))
        ent_nin.grid(row=0, column=1, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(popup, text="French Name:").grid(row=1, column=0, padx=20, pady=10, sticky="w")
        ent_fr = ctk.CTkEntry(popup)
        ent_fr.insert(0, extracted_data.get("french_name", "") or "")
        ent_fr.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(popup, text="Arabic Name:").grid(row=2, column=0, padx=20, pady=10, sticky="w")
        ent_ar = ctk.CTkEntry(popup)
        ent_ar.insert(0, extracted_data.get("arabic_name", "") or "")
        ent_ar.grid(row=2, column=1, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(popup, text="Category:").grid(row=3, column=0, padx=20, pady=10, sticky="w")
        category_var = ctk.StringVar(value="Visiteur")
        dropdown = ctk.CTkOptionMenu(popup, variable=category_var, values=["Apprenti longue durée", "Stagiaire", "Visiteur"], fg_color=ACCENT_BLUE, button_color=SIDEBAR_COLOR)
        dropdown.grid(row=3, column=1, padx=20, pady=10, sticky="ew")
        
        def save():
            nin = ent_nin.get()
            if not nin:
                messagebox.showerror("Error", "NIN is required")
                return
            db.create_user(nin, ent_fr.get(), ent_ar.get(), category_var.get())
            # Auto check-in upon creation
            db.log_access(nin, "CHECK_IN")
            popup.destroy()
            messagebox.showinfo("Success", "Profile created and Checked-In.")
            
        btn_save = ctk.CTkButton(popup, text="Confirm & Save", fg_color=SUCCESS_GREEN, hover_color="#146c43", command=save)
        btn_save.grid(row=4, column=0, columnspan=2, pady=30, ipadx=20, ipady=10)

    # --- WORKFLOW B: CHECK-IN / CHECK-OUT ---
    def show_checkin_popup(self, user):
        popup = ctk.CTkToplevel(self)
        popup.title("Member Found")
        popup.geometry("400x300")
        popup.grab_set()
        
        popup.grid_columnconfigure((0,1), weight=1)
        
        name = user['french_name'] or user['arabic_name'] or "Unknown"
        ctk.CTkLabel(popup, text=f"Member Found:\n{name}", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, columnspan=2, pady=30)
        
        last_action = db.get_last_action(user['nin'])
        
        def check_in():
            db.log_access(user['nin'], "CHECK_IN")
            popup.destroy()
            messagebox.showinfo("Success", "Checked In successfully.")
            
        def check_out():
            db.log_access(user['nin'], "CHECK_OUT")
            popup.destroy()
            messagebox.showinfo("Success", "Checked Out successfully.")
            
        btn_in = ctk.CTkButton(popup, text="Check-In", fg_color=SUCCESS_GREEN, hover_color="#146c43", height=50, font=ctk.CTkFont(size=16, weight="bold"), command=check_in)
        btn_in.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        
        btn_out = ctk.CTkButton(popup, text="Check-Out", fg_color=DANGER_RED, hover_color="#b02a37", height=50, font=ctk.CTkFont(size=16, weight="bold"), command=check_out)
        btn_out.grid(row=1, column=1, padx=20, pady=20, sticky="ew")
        
        if last_action == "CHECK_IN":
            btn_in.configure(state="disabled", fg_color="gray")
        else:
            # If never checked in, or currently checked out
            btn_out.configure(state="disabled", fg_color="gray")

    # --- PROFILES & LOGS ---
    def setup_profiles(self):
        self.profiles_frame.grid_columnconfigure(0, weight=1)
        self.profiles_frame.grid_rowconfigure(1, weight=1)
        
        lbl = ctk.CTkLabel(self.profiles_frame, text="Registered Profiles", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_COLOR_DARK)
        lbl.grid(row=0, column=0, pady=20, padx=20, sticky="w")
        
        self.scroll_frame = ctk.CTkScrollableFrame(self.profiles_frame, fg_color="transparent")
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
    def load_profiles(self):
        # Clear existing
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
            
        users = db.get_all_users()
        for i, u in enumerate(users):
            card = ctk.CTkFrame(self.scroll_frame, fg_color=CARD_COLOR, corner_radius=10)
            card.grid(row=i, column=0, sticky="ew", pady=5)
            card.grid_columnconfigure(1, weight=1)
            
            name = u['french_name'] or u['arabic_name'] or u['nin']
            ctk.CTkLabel(card, text=f"{name} ({u['category']})", font=ctk.CTkFont(weight="bold"), text_color=TEXT_COLOR_DARK).grid(row=0, column=0, padx=20, pady=10)
            
            btn_logs = ctk.CTkButton(card, text="View Logs", fg_color=ACCENT_BLUE, hover_color="#002244", command=lambda nin=u['nin']: self.show_user_logs(nin))
            btn_logs.grid(row=0, column=2, padx=20, pady=10)

    def show_user_logs(self, nin):
        popup = ctk.CTkToplevel(self)
        popup.title("User Logs")
        popup.geometry("500x400")
        
        scroll = ctk.CTkScrollableFrame(popup)
        scroll.pack(fill="both", expand=True, padx=20, pady=20)
        
        logs = db.get_user_logs(nin)
        if not logs:
            ctk.CTkLabel(scroll, text="No logs found.").pack(pady=20)
            return
            
        for log in logs:
            color = SUCCESS_GREEN if log['action'] == 'CHECK_IN' else DANGER_RED
            text = f"{log['action']} - {log['timestamp']}"
            lbl = ctk.CTkLabel(scroll, text=text, text_color=color, font=ctk.CTkFont(weight="bold"))
            lbl.pack(anchor="w", pady=5)

if __name__ == "__main__":
    app = DesktopApp()
    app.mainloop()
