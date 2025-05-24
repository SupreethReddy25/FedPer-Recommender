"""
Script to combine manager parts into a single file.
"""

import os

# Parts to combine
parts = [
    'manager.py',
    'manager_part2.py',
    'manager_part3.py',
    'manager_part4.py'
]

# Start with the base file content
with open('manager.py', 'r') as f:
    content = f.read()

# Remove the last line (presumably the class closing)
content = content.rstrip()

# Get the methods from other parts
for part_file in parts[1:]:
    with open(part_file, 'r') as f:
        part_content = f.read()
    
    # Extract functions only
    start = part_content.find('def ')
    if start != -1:
        part_content = part_content[start:]
        
        # Clean up indentation
        lines = part_content.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.startswith('def '):
                # Method definition, add indentation
                cleaned_lines.append('    ' + line)
            else:
                # Other lines, ensure proper indentation
                if line and not line.startswith('"""'):
                    cleaned_lines.append('    ' + line)
                else:
                    cleaned_lines.append(line)
        
        # Add to content
        content += '\n' + '\n'.join(cleaned_lines)

# Close the class
content += '\n'

# Write the combined file
with open('manager_combined.py', 'w') as f:
    f.write(content)

print("Combined manager file created: manager_combined.py")
