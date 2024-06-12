from collections import defaultdict

def build_org_structure(org_list):
    org_dict = defaultdict(list)
    all_employees = set()
    all_subordinates = set()
    
    # Build the org_dict and collect all employees and subordinates
    for report in org_list:
        manager = report[0]
        all_employees.add(manager)
        for employee in report[1:]:
            org_dict[manager].append(employee)
            all_employees.add(employee)
            all_subordinates.add(employee)
    
    # Find the root manager (employee who is not a subordinate)
    root_manager = list(all_employees - all_subordinates)[0]
    return org_dict, root_manager

def print_org_structure(org_dict, manager, level=0):
    print("  " * level + manager)
    for report in org_dict[manager]:
        print_org_structure(org_dict, report, level + 1)

def get_skip_level_reports(org_dict, manager):
    skip_level_reports = []
    for direct_report in org_dict[manager]:
        skip_level_reports.extend(org_dict[direct_report])
    return skip_level_reports

# Example usage:
org_list = [
    ["Alice", "Bob", "Charlie"],
    ["Bob", "David", "Eve"],
    ["Charlie", "Faythe", "Grace"]
]

org_dict, root_manager = build_org_structure(org_list)

# Print org structure
print("Org Structure:")
print_org_structure(org_dict, root_manager)

# Print skip level reports for a given manager
mgr_name = "Alice"
print(f"\nSkip level reports for {mgr_name}:")
skip_level_reports = get_skip_level_reports(org_dict, mgr_name)
print(skip_level_reports)

mgr_name = "Bob"
print(f"\nSkip level reports for {mgr_name}:")
skip_level_reports = get_skip_level_reports(org_dict, mgr_name)
print(skip_level_reports)
