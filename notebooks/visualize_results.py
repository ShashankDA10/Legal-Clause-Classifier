import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Clause mention status: 1 = Mentioned, 0 = Not mentioned
data = {
    'Clause': [
        'Agreement Date', 'Anti-Assignment', 'Document Name',
        'Effective Date', 'Expiration Date', 'Governing Law', 'Parties'
    ],
    'Mentioned': [
        0, 0, 0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Clause', y='Mentioned', palette="rocket")

plt.xticks(rotation=45)
plt.title('Clause Mention Status in Contract')
plt.xlabel('Clause Type')
plt.ylabel('Mentioned (1 = Yes, 0 = No)')
plt.tight_layout()
plt.show()
