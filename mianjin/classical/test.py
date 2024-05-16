import yaml


# Define a tuple
my_tuple = (1, 2, 3)

# Define a constructor for Python tuples
def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))

# Add the tuple constructor to the safe loader
yaml.SafeLoader.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)

# Save tuple to a YAML file
with open('tuple.yaml', 'w') as file:
    yaml.safe_dump(my_tuple, file)

# Reload tuple from the YAML file
with open('tuple.yaml', 'r') as file:
    reloaded_tuple = yaml.safe_load(file)

# Print reloaded tuple
print("Reloaded tuple:", reloaded_tuple)
