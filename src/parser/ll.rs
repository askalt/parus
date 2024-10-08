use std::marker::PhantomData;

use super::parser::{Parser, TreeNode};
use crate::{grammar::grammar::GrammarSymbol, lexer::lexer::Lexer, parser::parser::NonEpsTreeNode};

/// Describes LL(1) parser for the specific grammar.
pub struct LLParser<S: GrammarSymbol> {
    phantom: PhantomData<S>,
}

impl<S: GrammarSymbol> LLParser<S> {
    /// Create new LLParser.
    pub fn new() -> Self {
        // TODO: grammar validation.
        // Grammar should satisfy LL(1) laws.
        Self {
            phantom: PhantomData,
        }
    }

    /// Parse some non-terminal.
    fn parse_nt(&self, nt: usize, lexer: &mut dyn Lexer<S>) -> Option<Box<TreeNode<S>>> {
        let cur = lexer.cur();
        let productions = S::get_productions(nt);

        if cur.is_none() {
            // Check for the epsilon production.
            return if productions.iter().any(|it| it.is_none()) {
                Some(Box::new(TreeNode::NonEps(NonEpsTreeNode {
                    vertex: S::from_num(nt),
                    children: Some(vec![Box::new(TreeNode::Eps)]),
                })))
            } else {
                None
            };
        }

        let cur = cur.unwrap();

        // Will be true if `nt` has epsilon production.
        let mut has_eps = false;
        for production in productions.iter() {
            if production.is_none() {
                has_eps = true;
                continue;
            }
            let production = production.as_ref().unwrap();
            let mut children: Vec<Box<TreeNode<S>>> = Vec::new();

            let childs_to_skip = if S::is_terminal(production[0]) {
                if production[0] != cur.to_num() {
                    continue;
                }
                children.push(Box::new(TreeNode::NonEps(NonEpsTreeNode::<S> {
                    vertex: cur,
                    children: None,
                })));
                lexer.shift();
                1
            } else {
                0
            };

            // We just picked a right production and have no more variants.
            // There are 2 cases:
            // * We could pick production starting with terminal, then this terminal
            //   matched current symbol (and it's single production, because LL(1)).
            // * We could pick production that starting with nonterminal and
            //   there is must be single such production (else we can't
            //   choose only by single symbol lookup and it's not LL(1)).
            for s in production.iter().skip(childs_to_skip) {
                if S::is_terminal(*s) {
                    // Have terminal, try to match it with current symbol.
                    let cur = lexer.cur()?;
                    if *s != cur.to_num() {
                        // Can't match, nothing left to do, because we picked right production.
                        return None;
                    }
                    children.push(Box::new(TreeNode::NonEps(NonEpsTreeNode::<S> {
                        vertex: cur,
                        children: None,
                    })));
                    lexer.shift();
                } else {
                    // Have non-terminal, try parse it recursively.
                    let parsed = self.parse_nt(s.clone(), lexer)?;
                    children.push(parsed);
                }
            }
            return Some(Box::new(TreeNode::NonEps(NonEpsTreeNode::<S> {
                vertex: S::from_num(nt),
                children: Some(children),
            })));
        }
        // We didn't find any non-epsilon productions which match.
        if has_eps {
            // But we have epsilon production, so we can conclude that it is using here.
            // Again, the properties of LL(1) are used here.
            Some(Box::new(TreeNode::NonEps(NonEpsTreeNode {
                vertex: S::from_num(nt),
                children: Some(vec![Box::new(TreeNode::Eps)]),
            })))
        } else {
            None
        }
    }
}

impl<S: GrammarSymbol> Parser<S> for LLParser<S> {
    fn parse(&self, lexer: &mut dyn Lexer<S>) -> Option<Box<TreeNode<S>>> {
        self.parse_nt(S::start_non_terminal(), lexer)
    }
}
