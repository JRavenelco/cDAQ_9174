
import PyPDF2
import sys

def extract_text_from_pdf(pdf_path, start_page=0, num_pages=10):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_total_pages = len(reader.pages)
            print(f"Total pages: {num_total_pages}")
            
            text = ""
            end_page = min(start_page + num_pages, num_total_pages)
            
            for i in range(start_page, end_page):
                page = reader.pages[i]
                text += f"\n--- Page {i+1} ---\n"
                text += page.extract_text()
                
            return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

if __name__ == "__main__":
    pdf_path = r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\Pruebas\src\Tesis_doctoral.pdf"
    print(extract_text_from_pdf(pdf_path, start_page=25, num_pages=10))
