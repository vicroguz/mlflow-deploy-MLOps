.PHONY: venv install train validate clean

PY=python

venv:
	$(PY) -m venv .venv

install:
	. .venv/bin/activate || . .venv/Scripts/activate && pip install -U pip && pip install -r requirements.txt

train:
	$(PY) train.py

validate:
	$(PY) validate.py

clean:
	rm -rf mlruns model.pkl