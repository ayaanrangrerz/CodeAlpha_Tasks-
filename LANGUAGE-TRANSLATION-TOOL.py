import tkinter as tk
from tkinter import ttk

LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'French': 'fr',
    'Spanish': 'es',
}

try:
    from deep_translator import GoogleTranslator
except ModuleNotFoundError:
    GoogleTranslator = None


def translate_text():
    if GoogleTranslator is None:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "deep_translator is not installed.\nRun: pip install deep-translator")
        return

    text = input_text.get("1.0", tk.END).strip()
    if not text:
        return

    source = LANGUAGES.get(source_lang.get(), 'en')
    target = LANGUAGES.get(target_lang.get(), 'hi')

    translated = GoogleTranslator(
        source=source,
        target=target
    ).translate(text)

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, translated)

root = tk.Tk()
root.title("Language Translation Tool")
root.geometry("600x500")

tk.Label(root, text="Enter Text").pack()

input_text = tk.Text(root, height=8)
input_text.pack()

tk.Label(root, text="Source Language").pack()

source_lang = ttk.Combobox(root, state="readonly")
source_lang['values'] = tuple(LANGUAGES.keys())
source_lang.current(0)
source_lang.pack()

tk.Label(root, text="Target Language").pack()

target_lang = ttk.Combobox(root, state="readonly")
target_lang['values'] = tuple(LANGUAGES.keys())
target_lang.current(1)
target_lang.pack()

translate_button = tk.Button(root, text="Translate", command=translate_text)
translate_button.pack(pady=10)

tk.Label(root, text="Translated Text").pack()

output_text = tk.Text(root, height=8)
output_text.pack()

root.mainloop()