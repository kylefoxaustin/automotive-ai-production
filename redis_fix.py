# Fix Redis connection
with open('src/main.py', 'r') as f:
    content = f.read()

# Replace the Redis connection line
content = content.replace(
    "host=os.getenv('REDIS_HOST', 'localhost')",
    "host=os.getenv('REDIS_HOST', 'redis')"
)

with open('src/main.py', 'w') as f:
    f.write(content)

print("Redis connection fixed!")
