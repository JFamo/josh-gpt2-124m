import os

def aggregate_sent_items(root_folder, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as outfile:
        # Iterate through each subfolder in the root folder
        for subfolder in os.listdir(root_folder):
            subfolder_path = os.path.join(root_folder, subfolder)
            if os.path.isdir(subfolder_path):
                sent_items_path = os.path.join(subfolder_path, 'sent_items')
                # Check if the "sent_items" folder exists within the subfolder
                if os.path.isdir(sent_items_path):
                    # Iterate through each text file in the "sent_items" folder
                    for filename in os.listdir(sent_items_path):
                        file_path = os.path.join(sent_items_path, filename)
                        if os.path.isfile(file_path):
                            # Open each text file and write its contents to the output file
                            with open(file_path, 'r') as infile:
                                outfile.write(infile.read())
                                outfile.write("\n")  # Optionally add a newline between file contents

# Example usage
root_folder = 'C:/Users/joshu/Downloads/enron_mail_20150507/maildir'
output_file = './enron_data.txt'
aggregate_sent_items(root_folder, output_file)