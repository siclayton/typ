void DecisionTree::trainModel() {
    // A queue to hold the nodes we still need to calculate the feature and threshold for
    // Stores indexes of the nodes array
    int queue[MAX_NODES];
    int start = 0, end = 0;

    // Add the root node to the nodes array and the queue
    nodes[numNodes++] = {0, lenXTrain - 1, -1, -1, -1, -1, 0, false, -1};
    queue[end++] = 0;

    // Use the CART algorithm to create the tree
    // Loop while there are still nodes left in the queue
    while (start < end) {
        int currentIndex = queue[start++];
        TreeNode &current = nodes[currentIndex];

        // Stopping conditions
        if (current.depth >= MAX_DEPTH || current.end - current.start < MIN_SAMPLES_TO_SPLIT || nodeIsPure(current)) {
            current.isLeaf = true;
            current.prediction = findMajorityClass(current.start, current.end);
            continue;
        }

        Split bestSplit = findBestSplit(current.start, current.end);

        // Reorder indices array and find the index at which the values are split
        int midIndex = reorderIndices(current.start, current.end, bestSplit.feature, bestSplit.value);

        // Store the feature, value pair for the split at this node
        current.feature = bestSplit.feature;
        current.threshold = bestSplit.value;

        createChildren(current, queue, &end, midIndex);
    }
}
/**
 * Create the children for a given node and add them to the work queue
 * @param current the node to create the children of
 * @param queue the work queue
 * @param end a pointer to the end index of the array
 * @param midIndex the index of the indices array where the samples are split based on the feature,
 * threshold pair at the current node
 */
void DecisionTree::createChildren(TreeNode &current, int *queue, int *end, int midIndex) {
    int left = numNodes++;
    int right = numNodes++;
    current.left = left;
    current.right = right;
    nodes[left] = {current.start, midIndex, -1, -1, -1, -1, current.depth + 1, false, -1};
    nodes[right] = {midIndex + 1, current.end, -1, -1, -1, -1, current.depth + 1, false, -1};

    queue[(*end)++] = left;
    queue[(*end)++] = right;
}