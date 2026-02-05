try:
    with open(r'levitador valentin\adq_serie10ms.c', 'rb') as f:
        content = f.read()
        for encoding in ['latin-1', 'cp1252', 'utf-8', 'iso-8859-1']:
            try:
                text = content.decode(encoding)
                print(text)
                break
            except:
                continue
except Exception as e:
    print(f"Error: {e}")
