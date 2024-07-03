// Sugar to produce trees in tests.
// Need to define NonEpsNode above.
#[macro_export]
macro_rules! make_node {
    ($vertex: expr, $children: expr) => {
        Box::new(TreeNode::NonEps(NonEpsNode {
            vertex: $vertex,
            children: $children,
        }))
    };
}

#[macro_export]
macro_rules! eps_node {
    () => {
        Box::new(TreeNode::Eps)
    };
}
