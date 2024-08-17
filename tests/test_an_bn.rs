use std::cell::RefCell;

use grammar_derive::GrammarSymbol;
use parus::{
    grammar::grammar::{IterableGrammarSymbol, RandomGrammarIterator},
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
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug, GrammarSymbol)]
enum Node {
    #[to(
        A S B,
        Epsilon,
    )]
    S,

    A,
    B,
}

#[test]
fn iterate_over_grammar() {
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

    let actual: Vec<_> = Node::into_iterator()
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
    let mut lexer = AnBnLexer::from_bytes(expr.into());
    let parser: LLParser<Node> = LLParser::new();

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
    let mut lexer = AnBnLexer::from_bytes(expr.into());
    let parser: LLParser<Node> = LLParser::new();
    let res = parser.parse(&mut lexer);
    assert!(res.is_none());
}

#[test]
fn example_random_expressions() {
    let iterator: RandomGrammarIterator<Node> = RandomGrammarIterator::new(30, 35);
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
