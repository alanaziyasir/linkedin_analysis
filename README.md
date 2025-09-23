## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/alanaziyasir/linkedin_analysis.git
cd linkedin_analysis

Create a virtual environment (recommended)

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies

bashpip install -r requirements.txt
🚦 Running the Application
bash
python -m uvicorn app4_api:app --host 0.0.0.0 --port 8000 --reload

API will be available at http://localhost:8000
