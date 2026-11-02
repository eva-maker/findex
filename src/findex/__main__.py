from findex.cli import app

# Без цієї перевірки `python -m findex index --executor processes` не працював би:
# кожен воркер-процес (spawn) заново імпортує цей модуль і запустив би CLI ще раз.
if __name__ == "__main__":
    app(prog_name="findex")
