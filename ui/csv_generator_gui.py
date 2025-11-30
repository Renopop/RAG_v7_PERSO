"""
Générateur de CSV d'ingestion avec interface moderne CustomTkinter
Contourne la limitation des navigateurs web en accédant directement au système de fichiers
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter as tk
import csv
import os
from typing import List, Dict

# Import de la configuration des chemins
try:
    from config_manager import load_config
    _config = load_config()
    CSV_IMPORT_DIR = _config.csv_import_dir
except ImportError:
    # Fallback si config_manager n'est pas disponible
    CSV_IMPORT_DIR = r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\CSV_Ingestion"


class ToolTip:
    """
    Classe pour afficher des tooltips (infobulles) sur les widgets CustomTkinter
    """
    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.after_id = None

        # Bind events
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<Button-1>", self.on_leave)

    def on_enter(self, event=None):
        """Appelé quand la souris entre sur le widget"""
        self.after_id = self.widget.after(self.delay, self.show_tooltip)

    def on_leave(self, event=None):
        """Appelé quand la souris quitte le widget"""
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        self.hide_tooltip()

    def show_tooltip(self):
        """Affiche le tooltip"""
        if self.tooltip_window:
            return

        x, y, _, _ = self.widget.bbox("insert") if hasattr(self.widget, 'bbox') else (0, 0, 0, 0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        # Style du tooltip
        frame = tk.Frame(tw, background="#333333", borderwidth=1, relief="solid")
        frame.pack()

        label = tk.Label(
            frame,
            text=self.text,
            background="#333333",
            foreground="#ffffff",
            font=("Segoe UI", 9),
            padx=8,
            pady=4,
            wraplength=300,
            justify="left"
        )
        label.pack()

    def hide_tooltip(self):
        """Cache le tooltip"""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


# Configuration du thème
ctk.set_appearance_mode("dark")  # "dark", "light", ou "system"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"


class CSVGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuration de la fenêtre
        self.title("📋 Générateur de CSV d'Ingestion RAG")
        self.geometry("1100x750")  # Réduit de 850 à 750 pour meilleure compatibilité
        self.minsize(900, 600)  # Taille minimale si redimensionné

        # Liste des fichiers
        self.files_list: List[Dict] = []

        # Construction de l'interface
        self.setup_ui()

    def setup_ui(self):
        """Construit l'interface utilisateur"""

        # ====================
        # En-tête
        # ====================
        header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header_frame.pack(fill="x", padx=20, pady=(15, 5))  # Réduit les marges

        title_label = ctk.CTkLabel(
            header_frame,
            text="🎯 Générateur de CSV d'Ingestion",
            font=ctk.CTkFont(size=20, weight="bold")  # Réduit de 24 à 20
        )
        title_label.pack(side="left")

        info_label = ctk.CTkLabel(
            header_frame,
            text="✨ Vrais chemins de fichiers garantis",
            font=ctk.CTkFont(size=11),  # Réduit de 12 à 11
            text_color="gray"
        )
        info_label.pack(side="left", padx=15)  # Réduit de 20 à 15

        # ====================
        # Zone de contrôles
        # ====================
        controls_frame = ctk.CTkFrame(self)
        controls_frame.pack(fill="x", padx=20, pady=5)  # Réduit de 10 à 5

        # Bouton ajouter fichiers
        self.add_files_btn = ctk.CTkButton(
            controls_frame,
            text="📂 Ajouter fichiers",
            command=self.add_files,
            font=ctk.CTkFont(size=13, weight="bold"),  # Réduit de 14 à 13
            width=180,  # Réduit de 220 à 180
            height=35,  # Réduit de 40 à 35
            corner_radius=8
        )
        self.add_files_btn.pack(side="left", padx=8, pady=8)  # Réduit les marges
        ToolTip(self.add_files_btn, "Ouvre le sélecteur de fichiers pour ajouter des PDF, DOCX, TXT, etc. à la liste d'ingestion.")

        # Bouton scanner répertoire
        self.scan_dir_btn = ctk.CTkButton(
            controls_frame,
            text="📁 Scanner répertoire",
            command=self.scan_directory,
            font=ctk.CTkFont(size=13, weight="bold"),  # Réduit de 14 à 13
            width=180,  # Réduit de 220 à 180
            height=35,  # Réduit de 40 à 35
            corner_radius=8,
            fg_color="green",
            hover_color="darkgreen"
        )
        self.scan_dir_btn.pack(side="left", padx=8, pady=8)  # Réduit les marges
        ToolTip(self.scan_dir_btn, "Scanne un répertoire complet pour trouver automatiquement tous les fichiers supportés (PDF, DOCX, TXT, etc.).")

        # Groupe par défaut
        ctk.CTkLabel(
            controls_frame,
            text="Groupe:",
            font=ctk.CTkFont(size=11)  # Réduit de 12 à 11
        ).pack(side="left", padx=(15, 5))  # Réduit de 20 à 15

        self.default_group_entry = ctk.CTkEntry(
            controls_frame,
            placeholder_text="ALL",
            width=120,  # Réduit de 150 à 120
            height=35   # Réduit de 40 à 35
        )
        self.default_group_entry.pack(side="left", padx=5)
        self.default_group_entry.insert(0, "ALL")

        # Bouton appliquer groupe à tous
        apply_group_btn = ctk.CTkButton(
            controls_frame,
            text="Appliquer à tous",
            command=self.apply_default_group,
            width=130,  # Réduit de 150 à 130
            height=35   # Réduit de 40 à 35
        )
        apply_group_btn.pack(side="left", padx=8)  # Réduit de 10 à 8
        ToolTip(apply_group_btn, "Applique le groupe par défaut à tous les fichiers de la liste. Utile pour assigner rapidement une collection.")

        # ====================
        # Zone de liste des fichiers (scrollable)
        # ====================
        list_frame = ctk.CTkFrame(self)
        list_frame.pack(fill="both", expand=True, padx=20, pady=5)  # Réduit de 10 à 5

        # En-tête de la liste
        header = ctk.CTkFrame(list_frame, fg_color="gray25")
        header.pack(fill="x", padx=5, pady=3)  # Réduit de 5 à 3

        ctk.CTkLabel(
            header,
            text="Chemin du fichier",
            font=ctk.CTkFont(size=11, weight="bold"),  # Réduit de 12 à 11
            width=500,
            anchor="w"
        ).pack(side="left", padx=8)  # Réduit de 10 à 8

        ctk.CTkLabel(
            header,
            text="Groupe",
            font=ctk.CTkFont(size=11, weight="bold"),  # Réduit de 12 à 11
            width=150,
            anchor="w"
        ).pack(side="left", padx=8)  # Réduit de 10 à 8

        ctk.CTkLabel(
            header,
            text="",
            width=80
        ).pack(side="left")

        # Zone scrollable pour les fichiers - HAUTEUR RÉDUITE pour voir les boutons du bas
        self.scrollable_frame = ctk.CTkScrollableFrame(list_frame, height=250)  # Réduit de 400 à 250
        self.scrollable_frame.pack(fill="both", expand=True, padx=5, pady=3)  # Réduit de 5 à 3

        # Compteur
        self.count_label = ctk.CTkLabel(
            list_frame,
            text="0 fichier(s)",
            font=ctk.CTkFont(size=11),  # Réduit de 12 à 11
            text_color="gray"
        )
        self.count_label.pack(pady=3)  # Réduit de 5 à 3

        # ====================
        # Nom du CSV / Base de données
        # ====================
        csv_name_frame = ctk.CTkFrame(self, fg_color="transparent")
        csv_name_frame.pack(fill="x", padx=20, pady=(5, 3))  # Réduit les marges

        ctk.CTkLabel(
            csv_name_frame,
            text="💾 Nom du CSV (nom de base de données) :",
            font=ctk.CTkFont(size=12, weight="bold")  # Réduit de 13 à 12
        ).pack(anchor="w", pady=(0, 3))  # Réduit de 5 à 3

        csv_name_inner_frame = ctk.CTkFrame(csv_name_frame, fg_color="transparent")
        csv_name_inner_frame.pack(fill="x")

        self.csv_name_entry = ctk.CTkEntry(
            csv_name_inner_frame,
            placeholder_text="Ex: ma_base_easa, documents_techniques...",
            height=32,  # Réduit de 35 à 32
            font=ctk.CTkFont(size=11)  # Réduit de 12 à 11
        )
        self.csv_name_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))  # Réduit de 10 à 8

        ctk.CTkLabel(
            csv_name_inner_frame,
            text=".csv",
            font=ctk.CTkFont(size=11),  # Réduit de 12 à 11
            text_color="gray"
        ).pack(side="left")

        ctk.CTkLabel(
            csv_name_frame,
            text="💡 Ce nom sera utilisé pour créer la base FAISS",
            font=ctk.CTkFont(size=10),  # Réduit de 11 à 10
            text_color="gray60"
        ).pack(anchor="w", pady=(2, 0))  # Réduit de 3 à 2

        # ====================
        # Boutons d'action - COMPACTS pour garantir la visibilité
        # ====================
        action_frame = ctk.CTkFrame(self, fg_color="transparent")
        action_frame.pack(fill="x", padx=20, pady=(5, 15))  # Réduit les marges

        # Bouton ouvrir CSV existant
        open_csv_btn = ctk.CTkButton(
            action_frame,
            text="📂 Ouvrir CSV",
            command=self.open_csv,
            fg_color="blue",
            hover_color="darkblue",
            width=150,  # Réduit de 180 à 150
            height=38   # Réduit de 45 à 38
        )
        open_csv_btn.pack(side="left", padx=8)  # Réduit de 10 à 8
        ToolTip(open_csv_btn, "Ouvre un CSV existant pour le modifier. Permet d'ajouter ou supprimer des fichiers d'une liste existante.")

        # Bouton effacer tout
        clear_btn = ctk.CTkButton(
            action_frame,
            text="🗑️ Effacer tout",
            command=self.clear_all,
            fg_color="gray30",
            hover_color="gray40",
            width=150,  # Réduit de 180 à 150
            height=38   # Réduit de 45 à 38
        )
        clear_btn.pack(side="left", padx=8)  # Réduit de 10 à 8
        ToolTip(clear_btn, "Supprime tous les fichiers de la liste. Demande une confirmation avant de vider la liste.")

        # Bouton sauvegarder CSV - LE PLUS IMPORTANT
        self.generate_btn = ctk.CTkButton(
            action_frame,
            text="💾 Sauvegarder CSV",
            command=self.generate_csv,
            font=ctk.CTkFont(size=14, weight="bold"),  # Réduit de 16 à 14
            width=180,  # Réduit de 200 à 180
            height=38,  # Réduit de 45 à 38
            corner_radius=8
        )
        self.generate_btn.pack(side="right", padx=8)  # Réduit de 10 à 8
        ToolTip(self.generate_btn, "Génère le fichier CSV avec tous les fichiers de la liste. Le nom du CSV devient le nom de la base FAISS lors de l'ingestion.")

    def add_files(self):
        """Ouvre le file dialog et ajoute les fichiers sélectionnés"""
        filetypes = (
            ("Tous les fichiers supportés", "*.pdf *.docx *.doc *.txt *.md *.csv *.xml"),
            ("PDF", "*.pdf"),
            ("Word", "*.docx *.doc"),
            ("Texte", "*.txt *.md"),
            ("CSV", "*.csv"),
            ("XML", "*.xml"),
            ("Tous", "*.*")
        )

        filenames = filedialog.askopenfilenames(
            title="Sélectionnez des fichiers",
            filetypes=filetypes
        )

        if filenames:
            default_group = self.default_group_entry.get() or "ALL"

            for filepath in filenames:
                # Vérifier si déjà dans la liste
                if not any(f["path"] == filepath for f in self.files_list):
                    self.files_list.append({
                        "path": filepath,
                        "group": default_group
                    })

            self.refresh_file_list()

    def scan_directory(self):
        """Ouvre une fenêtre de dialogue pour scanner un répertoire"""
        # Fenêtre de configuration du scan (taille agrandie)
        scan_window = ctk.CTkToplevel(self)
        scan_window.title("📁 Scanner un répertoire")
        scan_window.geometry("900x550")
        scan_window.transient(self)
        scan_window.grab_set()

        # Centrer la fenêtre
        scan_window.update_idletasks()
        x = (scan_window.winfo_screenwidth() // 2) - (900 // 2)
        y = (scan_window.winfo_screenheight() // 2) - (550 // 2)
        scan_window.geometry(f"900x550+{x}+{y}")

        # Contenu
        main_frame = ctk.CTkFrame(scan_window)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Titre
        ctk.CTkLabel(
            main_frame,
            text="📁 Configuration du scan de répertoire",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(0, 20))

        # Sélection du répertoire
        dir_frame = ctk.CTkFrame(main_frame)
        dir_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            dir_frame,
            text="Répertoire à scanner :",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)

        dir_select_frame = ctk.CTkFrame(dir_frame)
        dir_select_frame.pack(fill="x", padx=10, pady=5)

        dir_entry = ctk.CTkEntry(
            dir_select_frame,
            placeholder_text="Aucun répertoire sélectionné",
            width=650
        )
        dir_entry.pack(side="left", padx=5, fill="x", expand=True)

        def browse_directory():
            directory = filedialog.askdirectory(title="Sélectionnez un répertoire")
            if directory:
                dir_entry.delete(0, "end")
                dir_entry.insert(0, directory)

        ctk.CTkButton(
            dir_select_frame,
            text="📂 Parcourir",
            command=browse_directory,
            width=100
        ).pack(side="left", padx=5)

        # Options
        options_frame = ctk.CTkFrame(main_frame)
        options_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            options_frame,
            text="Options :",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)

        # Checkbox récursif
        recursive_var = ctk.BooleanVar(value=True)
        recursive_check = ctk.CTkCheckBox(
            options_frame,
            text="Scanner les sous-dossiers (récursif)",
            variable=recursive_var,
            font=ctk.CTkFont(size=12)
        )
        recursive_check.pack(anchor="w", padx=20, pady=5)

        # Filtres extensions
        ext_frame = ctk.CTkFrame(main_frame)
        ext_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            ext_frame,
            text="Extensions à inclure :",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w", padx=10, pady=5)

        # Checkboxes pour extensions
        ext_vars = {}
        extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".xml"]
        ext_checks_frame = ctk.CTkFrame(ext_frame)
        ext_checks_frame.pack(fill="x", padx=20, pady=5)

        for i, ext in enumerate(extensions):
            var = ctk.BooleanVar(value=True)
            ext_vars[ext] = var
            check = ctk.CTkCheckBox(
                ext_checks_frame,
                text=ext,
                variable=var,
                font=ctk.CTkFont(size=11)
            )
            # 3 colonnes
            row = i // 3
            col = i % 3
            check.grid(row=row, column=col, padx=10, pady=3, sticky="w")

        # Boutons
        buttons_frame = ctk.CTkFrame(main_frame)
        buttons_frame.pack(fill="x", pady=20)

        def do_scan():
            directory = dir_entry.get()
            if not directory or not os.path.isdir(directory):
                messagebox.showerror("Erreur", "Veuillez sélectionner un répertoire valide.")
                return

            recursive = recursive_var.get()
            selected_extensions = [ext for ext, var in ext_vars.items() if var.get()]

            if not selected_extensions:
                messagebox.showwarning("Attention", "Veuillez sélectionner au moins une extension.")
                return

            # Scanner le répertoire
            found_files = []
            try:
                if recursive:
                    for root, dirs, files in os.walk(directory):
                        for file in files:
                            ext = os.path.splitext(file)[1].lower()
                            if ext in selected_extensions:
                                full_path = os.path.join(root, file)
                                found_files.append(full_path)
                else:
                    for file in os.listdir(directory):
                        full_path = os.path.join(directory, file)
                        if os.path.isfile(full_path):
                            ext = os.path.splitext(file)[1].lower()
                            if ext in selected_extensions:
                                found_files.append(full_path)

                # Ajouter les fichiers trouvés
                default_group = self.default_group_entry.get() or "ALL"
                added_count = 0

                for filepath in sorted(found_files):
                    # Vérifier si déjà dans la liste
                    if not any(f["path"] == filepath for f in self.files_list):
                        self.files_list.append({
                            "path": filepath,
                            "group": default_group
                        })
                        added_count += 1

                scan_window.destroy()
                self.refresh_file_list()

                messagebox.showinfo(
                    "Scan terminé",
                    f"✅ {len(found_files)} fichier(s) trouvé(s)\n"
                    f"📥 {added_count} fichier(s) ajouté(s)\n"
                    f"⏭️ {len(found_files) - added_count} déjà présent(s)"
                )

            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors du scan :\n{e}")

        ctk.CTkButton(
            buttons_frame,
            text="🔍 Scanner",
            command=do_scan,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            width=150,
            fg_color="green",
            hover_color="darkgreen"
        ).pack(side="right", padx=10)

        ctk.CTkButton(
            buttons_frame,
            text="Annuler",
            command=scan_window.destroy,
            height=40,
            width=100,
            fg_color="gray",
            hover_color="darkgray"
        ).pack(side="right", padx=10)

    def refresh_file_list(self):
        """Rafraîchit l'affichage de la liste des fichiers (utilisé uniquement au chargement)"""
        # Nettoyer la frame scrollable
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Réinitialiser la liste des widgets
        self.file_widgets = []

        # Recréer les entrées
        for idx, file_data in enumerate(self.files_list):
            self.create_file_entry(idx, file_data)

        # Mettre à jour le compteur
        self.update_counter()

    def update_counter(self):
        """Met à jour le compteur de fichiers"""
        self.count_label.configure(text=f"{len(self.files_list)} fichier(s)")

    def create_file_entry(self, idx: int, file_data: Dict):
        """Crée une entrée de fichier dans la liste"""
        entry_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="gray20")
        entry_frame.pack(fill="x", padx=3, pady=3)

        # Chemin (lecture seule)
        path_label = ctk.CTkLabel(
            entry_frame,
            text=file_data["path"],
            font=ctk.CTkFont(size=10),
            width=500,
            anchor="w"
        )
        path_label.pack(side="left", padx=8, pady=6)

        # Groupe (éditable)
        group_entry = ctk.CTkEntry(
            entry_frame,
            width=150,
            height=30
        )
        group_entry.insert(0, file_data["group"])
        group_entry.pack(side="left", padx=8)

        # Stocker la référence pour mise à jour directe
        widget_info = {
            "frame": entry_frame,
            "group_entry": group_entry,
            "file_data": file_data  # Référence directe aux données
        }

        # S'assurer que la liste existe
        if not hasattr(self, 'file_widgets'):
            self.file_widgets = []
        self.file_widgets.append(widget_info)

        # Callback pour mettre à jour le groupe (référence directe, pas d'index)
        def update_group(event=None):
            file_data["group"] = group_entry.get()

        group_entry.bind("<KeyRelease>", update_group)

        # Bouton supprimer (utilise la référence widget_info)
        remove_btn = ctk.CTkButton(
            entry_frame,
            text="✕",
            command=lambda wi=widget_info: self.remove_file_widget(wi),
            width=60,
            height=30,
            fg_color="red",
            hover_color="darkred"
        )
        remove_btn.pack(side="left", padx=8)

    def remove_file_widget(self, widget_info: Dict):
        """Supprime un fichier via sa référence widget (sans rebuild)"""
        # Supprimer des données
        if widget_info["file_data"] in self.files_list:
            self.files_list.remove(widget_info["file_data"])

        # Supprimer de la liste des widgets
        if widget_info in self.file_widgets:
            self.file_widgets.remove(widget_info)

        # Détruire uniquement ce widget
        widget_info["frame"].destroy()

        # Mettre à jour le compteur
        self.update_counter()

    def remove_file(self, idx: int):
        """Supprime un fichier de la liste (compatibilité)"""
        if not hasattr(self, 'file_widgets'):
            return
        if 0 <= idx < len(self.file_widgets):
            self.remove_file_widget(self.file_widgets[idx])

    def clear_all(self):
        """Efface tous les fichiers"""
        if self.files_list:
            if messagebox.askyesno("Confirmation", "Voulez-vous vraiment effacer tous les fichiers ?"):
                # Détruire tous les widgets d'un coup
                for widget in self.scrollable_frame.winfo_children():
                    widget.destroy()
                self.files_list.clear()
                self.file_widgets = []
                self.update_counter()

    def apply_default_group(self):
        """Applique le groupe par défaut à tous les fichiers (sans rebuild)"""
        if not hasattr(self, 'file_widgets'):
            return
        default_group = self.default_group_entry.get() or "ALL"

        # Mettre à jour les données ET les widgets directement
        for widget_info in self.file_widgets:
            widget_info["file_data"]["group"] = default_group
            # Mettre à jour l'affichage du champ groupe
            group_entry = widget_info["group_entry"]
            group_entry.delete(0, "end")
            group_entry.insert(0, default_group)

    def generate_csv(self):
        """Génère et sauvegarde le CSV"""
        if not self.files_list:
            messagebox.showwarning("Aucun fichier", "Veuillez ajouter au moins un fichier.")
            return

        # Vérifier qu'un nom de CSV a été saisi
        csv_name = self.csv_name_entry.get().strip()
        if not csv_name:
            messagebox.showwarning(
                "Nom manquant",
                "Veuillez saisir un nom pour le fichier CSV.\n\n"
                "Ce nom sera utilisé comme nom de base de données lors de l'ingestion."
            )
            self.csv_name_entry.focus()
            return

        # S'assurer que le nom ne contient pas de caractères invalides
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in csv_name for char in invalid_chars):
            messagebox.showerror(
                "Nom invalide",
                f"Le nom ne doit pas contenir les caractères suivants :\n{' '.join(invalid_chars)}"
            )
            return

        # S'assurer que le répertoire existe
        os.makedirs(CSV_IMPORT_DIR, exist_ok=True)

        # Construire le chemin complet dans CSV_IMPORT_DIR
        csv_path = os.path.join(CSV_IMPORT_DIR, f"{csv_name}.csv")

        # Vérifier si le fichier existe déjà et demander confirmation
        if os.path.exists(csv_path):
            if not messagebox.askyesno(
                "Fichier existant",
                f"Le fichier '{csv_name}.csv' existe déjà.\n\n"
                f"Voulez-vous l'écraser ?"
            ):
                return

        try:
            # Générer le CSV
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["group", "path"])

                for file_data in self.files_list:
                    writer.writerow([file_data["group"], file_data["path"]])

            # Extraire le nom du fichier sans extension pour afficher le nom de la base
            csv_filename = os.path.basename(csv_path)
            db_name = os.path.splitext(csv_filename)[0]

            messagebox.showinfo(
                "Succès",
                f"CSV généré avec succès !\n\n"
                f"📁 Fichier : {csv_path}\n"
                f"✅ Fichiers : {len(self.files_list)}\n"
                f"🗄️ Nom de base : {db_name}\n\n"
                f"Utilisez ce CSV dans l'onglet 'Ingestion de documents' de Streamlit.\n"
                f"La base FAISS sera créée avec le nom : {db_name}"
            )

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la génération du CSV :\n{e}")

    def load_csv_from_path(self, csv_path: str):
        """Charge un CSV depuis un chemin spécifié (utilisé au démarrage avec argument)"""
        if not csv_path or not os.path.exists(csv_path):
            return

        try:
            # Lire le CSV
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f, delimiter=";")
                rows = list(reader)

            if not rows:
                messagebox.showerror("Erreur", "Le CSV est vide.")
                return

            # Détecter l'en-tête
            header = [h.strip().lstrip("\ufeff").lower() for h in rows[0]]

            if "group" in header and "path" in header:
                idx_group = header.index("group")
                idx_path = header.index("path")
                data_rows = rows[1:]
            else:
                # Pas d'en-tête détecté, on suppose group;path
                idx_group = 0
                idx_path = 1 if len(rows[0]) > 1 else 0
                data_rows = rows

            # Charger les données
            loaded_count = 0

            for row in data_rows:
                if len(row) <= max(idx_group, idx_path):
                    continue

                group = row[idx_group].strip()
                path = row[idx_path].strip()

                if not path:
                    continue

                # Ajouter à la liste (pas de vérification de doublon pour le chargement initial)
                self.files_list.append({
                    "path": path,
                    "group": group or "ALL"
                })
                loaded_count += 1

            self.refresh_file_list()

            # Pré-remplir le nom du CSV dans le champ
            csv_filename = os.path.basename(csv_path)
            csv_name = os.path.splitext(csv_filename)[0]
            self.csv_name_entry.delete(0, "end")
            self.csv_name_entry.insert(0, csv_name)

            messagebox.showinfo(
                "CSV chargé",
                f"✅ CSV ouvert avec succès !\n\n"
                f"📁 {csv_path}\n"
                f"📥 {loaded_count} fichier(s) chargé(s)\n\n"
                f"Vous pouvez maintenant ajouter de nouveaux fichiers\n"
                f"ou scanner un répertoire, puis sauvegarder le CSV."
            )

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'ouverture du CSV :\n{e}")

    def open_csv(self):
        """Ouvre et charge un CSV existant"""
        # S'assurer que le répertoire existe
        os.makedirs(CSV_IMPORT_DIR, exist_ok=True)

        # Demander quel CSV ouvrir (commence dans CSV_IMPORT_DIR)
        csv_path = filedialog.askopenfilename(
            title="Ouvrir un CSV existant",
            initialdir=CSV_IMPORT_DIR,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not csv_path:
            return

        try:
            # Lire le CSV
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f, delimiter=";")
                rows = list(reader)

            if not rows:
                messagebox.showerror("Erreur", "Le CSV est vide.")
                return

            # Détecter l'en-tête
            header = [h.strip().lstrip("\ufeff").lower() for h in rows[0]]

            if "group" in header and "path" in header:
                idx_group = header.index("group")
                idx_path = header.index("path")
                data_rows = rows[1:]
            else:
                # Pas d'en-tête détecté, on suppose group;path
                idx_group = 0
                idx_path = 1 if len(rows[0]) > 1 else 0
                data_rows = rows

            # Charger les données
            loaded_count = 0
            already_present = 0

            for row in data_rows:
                if len(row) <= max(idx_group, idx_path):
                    continue

                group = row[idx_group].strip()
                path = row[idx_path].strip()

                if not path:
                    continue

                # Vérifier si déjà dans la liste
                if any(f["path"] == path for f in self.files_list):
                    already_present += 1
                    continue

                self.files_list.append({
                    "path": path,
                    "group": group or "ALL"
                })
                loaded_count += 1

            self.refresh_file_list()

            # Pré-remplir le nom du CSV dans le champ
            csv_filename = os.path.basename(csv_path)
            csv_name = os.path.splitext(csv_filename)[0]
            self.csv_name_entry.delete(0, "end")
            self.csv_name_entry.insert(0, csv_name)

            messagebox.showinfo(
                "CSV chargé",
                f"✅ CSV ouvert avec succès !\n\n"
                f"📁 {csv_filename}\n"
                f"📥 {loaded_count} fichier(s) chargé(s)\n"
                f"⏭️ {already_present} déjà présent(s)\n\n"
                f"Vous pouvez maintenant ajouter de nouveaux fichiers\n"
                f"ou scanner un répertoire, puis sauvegarder le CSV."
            )

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'ouverture du CSV :\n{e}")


def main():
    """Point d'entrée principal"""
    import sys

    app = CSVGeneratorApp()

    # Si un CSV est fourni en argument, le charger automatiquement
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if os.path.exists(csv_path) and csv_path.endswith('.csv'):
            app.after(100, lambda: app.load_csv_from_path(csv_path))

    app.mainloop()


if __name__ == "__main__":
    main()
