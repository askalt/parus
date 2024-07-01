use crate::grammar::grammar::Symbol;
use crate::lexer::lexer::Lexer;

/// Describes non epsilon tree node.
#[derive(Debug, PartialEq, Eq)]
pub struct NonEpsTreeNode<S: Symbol> {
    // Current node symbol.
    pub vertex: S,
    // Children nodes.
    // It's `None` if the vertex is terminal.
    pub children: Option<Vec<Box<TreeNode<S>>>>,
}

/// Describes parsing tree.
#[derive(Debug, PartialEq, Eq)]
pub enum TreeNode<S: Symbol> {
    NonEps(NonEpsTreeNode<S>),
    Eps,
}

/// Describes parser for the specific grammar.
pub trait Parser<S: Symbol> {
    /// Parse grammar using the lexer.
    fn parse(&self, lexer: &mut dyn Lexer<S>) -> Option<Box<TreeNode<S>>>;
}
