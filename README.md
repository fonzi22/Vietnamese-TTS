# Prepare
- Use GPU for this.
## Set virtual environment:
- Linux:
```bash
python3 -m venv myenv
source myenv/bin/activate
```
- Window(Run as Administrator if need):
```bash
python -m venv myenv
myenv\Scripts\activate
```
## Setup commands:
```bash
python setup.py
```
## Run text to speech model:
- If you get the bug in the first try, because there are some bulding module in the first time you run this file. Please try it again!
- A text can have many sentences. But, a sentence should be no more than 250 chracters.
```bash
python main.py
```