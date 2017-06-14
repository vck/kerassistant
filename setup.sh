echo "setting up database..."
python db.py

echo "setting up env..."
mkdir models

echo "done..."

echo "running assistant..."
python mlbot.py

