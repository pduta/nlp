import re

text = """
Contact john.doe@example.com or jane_smith@company.org for help.
Follow us at #Python #MachineLearning #AI on social media.
Meeting scheduled on 2024-01-15 and follow-up on 15/03/2024.
Call us at +1-800-555-1234 or (415) 123-4567 or 9876543210.
"""

# Email usernames (part before @)
email_usernames = re.findall(r'([\w.]+)@[\w.]+', text)

# Hashtags
hashtags = re.findall(r'#(\w+)', text)

# Dates (YYYY-MM-DD and DD/MM/YYYY)
dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b', text)

# Phone numbers
phones = re.findall(r'\+?[\d][(\d\s\-().]{7,}\d', text)

print("Email Usernames:", email_usernames)
print("Hashtags:       ", hashtags)
print("Dates:          ", dates)
print("Phone Numbers:  ", phones)
