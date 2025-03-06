#Order goes:

# List of processing steps
steps = [
    "[0] - adTree_InitialPlyToDF.py",
    "[1] - adTree_ClusterInitialDF.py", 
    "[2] - adTree_voxelise.py",
    "[3] - adTree_AssignLargerClusters.py",
    "[4] - adTree_assignResources.py"
]

# Print the list of steps
print("Available processing steps:")
for step in steps:
    print(step)

# Ask user which step to start from
user_choice = int(input("Enter the number of the step to start from: "))

if user_choice == 0:
    print(f'starting from {steps[user_choice]}')
    import adTree_InitialPlyToDF
elif user_choice <= 1:
    print(f'starting from {steps[user_choice]}')
    import adTree_ClusterInitialDF
elif user_choice <= 2:
    print(f'starting from {steps[user_choice]}')
    import adTree_voxelise
elif user_choice <=3:
    print(f'starting from {steps[user_choice]}')
    import adTree_AssignLargerClusters
elif user_choice <= 4:
    print(f'starting from {steps[user_choice]}')
    import adTree_AssignResources
    adTree_AssignResources.main()

#adTree_InitialPlyToDF.py
# This converts the ply file to an initial qsm dataframe

#adTree_ClusterInitialDF.py
# This clusters the initial qsm dataframe into clusters, segments

#adTree_voxelise.py
# This voxelises the clusters, segments

#adTree_assignResources.py
# This assigns resources to the voxelised clusters, segments