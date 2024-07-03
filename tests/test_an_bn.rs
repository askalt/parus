use std::cell::RefCell;

use either::Either;
use parus::{
    grammar::grammar::{Epsilon, Grammar, RandomGrammarIterator, Symbol},
    lexer::lexer::Lexer,
    parser::{
        ll::LLParser,
        parser::{NonEpsTreeNode, Parser, TreeNode},
    },
};

/// a^n b^n grammar
/// V = {S, A, B}
///
/// S -> A S B | eps
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
enum Node {
    S,
    A,
    B,
}

impl Symbol for Node {
    fn is_terminal(&self) -> bool {
        matches!(self, Self::A | Self::B)
    }

    fn start_non_terminal() -> Self {
        Self::S
    }

    fn is_accept(&self, oth: &Self) -> bool {
        match self.is_terminal() {
            true => self == oth,
            _ => false,
        }
    }
}

struct AnBnGrammar {}

impl AnBnGrammar {
    const PRODUCTIONS: [&'static [Either<&'static [Node], Epsilon>]; 1] = [&[
        // S -> A S B
        Either::Left(&[Node::A, Node::S, Node::B]),
        // S -> eps
        Either::Right(Epsilon {}),
    ]];
}

impl Grammar<Node> for AnBnGrammar {
    fn get_productions(&self, symbol: &Node) -> &[Either<&[Node], Epsilon>] {
        match symbol {
            Node::S => Self::PRODUCTIONS[0],
            _ => panic!("only non-terminals have productions"),
        }
    }
}

#[test]
fn iterate_over_grammar() {
    let grammar = AnBnGrammar {};
    let expected = vec![
        "",
        "AB",
        "AABB",
        "AAABBB",
        "AAAABBBB",
        "AAAAABBBBB",
        "AAAAAABBBBBB",
        "AAAAAAABBBBBBB",
        "AAAAAAAABBBBBBBB",
        "AAAAAAAAABBBBBBBBB",
    ];

    let actual: Vec<_> = grammar
        .into_iterator()
        .take(expected.len())
        .map(|str| {
            str.iter()
                .map(|it| format!("{:?}", it))
                .collect::<Vec<String>>()
                .join("")
        })
        .collect();
    assert_eq!(actual, expected);
}

struct AnBnLexer {
    bytes: Vec<u8>,
    pos: usize,
    cur: RefCell<Option<Node>>,
}

impl AnBnLexer {
    fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes: bytes,
            pos: 0,
            cur: RefCell::new(None),
        }
    }
}

impl Lexer<Node> for AnBnLexer {
    fn cur(&self) -> Option<Node> {
        if self.pos == self.bytes.len() {
            return None;
        }
        if let Some(ok) = self.cur.borrow().as_ref() {
            return Some(ok.clone());
        }
        let cur = match self.bytes[self.pos] {
            b'A' => Node::A,
            b'B' => Node::B,
            _ => panic!("unexpected next symbol"),
        };
        self.cur.replace(Some(cur.clone()));
        Some(cur)
    }

    fn shift(&mut self) {
        self.pos += 1;
        self.cur.replace(None);
    }
}

mod util;

type NonEpsNode = NonEpsTreeNode<Node>;

#[test]
fn simple_expr_ok() {
    let expr = b"AAABBB";
    let grammar = AnBnGrammar {};
    let mut lexer = AnBnLexer::from_bytes(expr.into());
    let parser: LLParser<Node, AnBnGrammar> = LLParser::new(grammar);

    let tree = parser.parse(&mut lexer);
    assert_eq!(
        tree,
        Some(make_node!(
            Node::S,
            Some(vec![
                make_node!(Node::A, None),
                make_node!(
                    Node::S,
                    Some(vec![
                        make_node!(Node::A, None),
                        make_node!(
                            Node::S,
                            Some(vec![
                                make_node!(Node::A, None),
                                make_node!(Node::S, Some(vec![eps_node!()])),
                                make_node!(Node::B, None),
                            ])
                        ),
                        make_node!(Node::B, None),
                    ])
                ),
                make_node!(Node::B, None)
            ])
        ))
    );
}

#[test]
fn simple_expr_not_ok() {
    let expr = b"AAB";
    let grammar = AnBnGrammar {};
    let mut lexer = AnBnLexer::from_bytes(expr.into());
    let parser: LLParser<Node, AnBnGrammar> = LLParser::new(grammar);
    let res = parser.parse(&mut lexer);
    assert!(res.is_none());
}

#[test]
fn example_random_expressions() {
    let grammar = AnBnGrammar {};
    let iterator = RandomGrammarIterator::new(grammar, 30, 35);
    let actual: Vec<_> = iterator
        .take(10)
        .map(|str| {
            str.iter()
                .map(|it| format!("{:?}", it))
                .collect::<Vec<String>>()
                .join("")
        })
        .collect();
    println!("{:?}", actual.iter().map(|it| it.len()).collect::<Vec<_>>());
    println!("{}", actual.join("\n"));
}
