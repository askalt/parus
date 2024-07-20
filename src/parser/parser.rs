use crate::grammar::grammar::GrammarSymbol;
use crate::lexer::lexer::Lexer;

/// Describes non epsilon tree node.
#[derive(Debug, PartialEq, Eq)]
pub struct NonEpsTreeNode<S: GrammarSymbol> {
    // Current node symbol.
    pub vertex: S,
    // Children nodes.
    // It's `None` if the vertex is terminal.
    // Ideally, we want to have here enum of compile-time known structures,
    // one structure for each S production rule.
    pub children: Option<Vec<Box<TreeNode<S>>>>,
}

/// Describes parsing tree.
#[derive(Debug, PartialEq, Eq)]
pub enum TreeNode<S: GrammarSymbol> {
    NonEps(NonEpsTreeNode<S>),
    Eps,
}

impl<S: GrammarSymbol> TreeNode<S> {
    pub fn is_eps(&self) -> bool {
        matches!(self, Self::Eps)
    }

    pub fn is_non_eps(&self) -> bool {
        !self.is_eps()
    }

    pub fn as_non_eps(&self) -> Option<&NonEpsTreeNode<S>> {
        match self {
            Self::NonEps(x) => Some(x),
            _ => None,
        }
    }
}

/// Describes parser for the specific grammar.
pub trait Parser<S: GrammarSymbol> {
    /// Parse grammar using the lexer.
    fn parse(&self, lexer: &mut dyn Lexer<S>) -> Option<Box<TreeNode<S>>>;
}
