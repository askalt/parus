use core::panic;
use std::collections::HashMap;

use either::Either::{self};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DataEnum, DeriveInput, Ident, Token};

const EPSILON_IDENT: &str = "Epsilon";

struct Epsilon;

/// List of productions separated by commas.
/// Each production has one of the following forms:
/// - ident1 ident2 ... identN
///        where each ident is non Epsilon.
/// - Epsilon
struct Productions {
    elems: Vec<Either<Vec<syn::Ident>, Epsilon>>,
}

impl syn::parse::Parse for Productions {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut productions: Vec<either::Either<Vec<syn::Ident>, Epsilon>> = vec![];
        let mut flush = |production: &Vec<syn::Ident>| {
            if production[0] == EPSILON_IDENT {
                if production.len() > 1 {
                    return Err(syn::Error::new(
                        input.span(),
                        "Epsilon must be single ident on the production right side",
                    ));
                }
                productions.push(Either::Right(Epsilon {}));
            } else {
                productions.push(Either::Left(production.clone()));
            };
            Ok(())
        };

        let mut production: Vec<syn::Ident> = vec![];
        while !input.is_empty() {
            if input.peek(Token![,]) {
                input.parse::<proc_macro2::Punct>()?;
                flush(&production)?;
                production.clear();
                continue;
            }
            let id = input.parse::<syn::Ident>()?;
            if id == EPSILON_IDENT && production.len() > 0 {
                return Err(syn::Error::new(
                    input.span(),
                    "Epsilon must be single ident on the production right side",
                ));
            }
            production.push(id);
        }

        if !production.is_empty() {
            flush(&production)?;
        }

        Ok(Self { elems: productions })
    }
}

/// Idents are separated by commas.
struct IdentList {
    elems: Vec<syn::Ident>,
}

impl syn::parse::Parse for IdentList {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut elems = vec![];
        while !input.is_empty() {
            if input.peek(Token![,]) {
                input.parse::<proc_macro2::Punct>()?;
            }
            let nxt = input.parse::<syn::Ident>()?;
            elems.push(nxt);
        }
        Ok(Self { elems: elems })
    }
}

struct Variant {
    name: syn::Ident,
    // Option<Vec<usize>> is None iff it's Epsilon production.
    productions: Option<Vec<Option<Vec<usize>>>>,
    generator: Option<Ident>,
    fields_count: usize,
}

/// Extract variant list from the enum.
///
/// Panics if something is wrong.
fn extract_variants(data: &DataEnum) -> Vec<Variant> {
    let mut index_by_variant = HashMap::<syn::Ident, usize>::new();
    for (i, variant) in data.variants.iter().enumerate() {
        index_by_variant.insert(variant.ident.clone(), i);
    }

    let mut variants = vec![];
    for variant in data.variants.iter() {
        if variant.fields.len() > 1 {
            // TODO: maybe it works?
            panic!("Variant can't have more than 1 field");
        }
        match &variant.fields {
            // TODO: maybe it works?
            syn::Fields::Named(_) => panic!("Named variant fields not supported"),
            _ => {}
        };
        if variant.ident == EPSILON_IDENT {
            panic!("{} is a reserved keyword", EPSILON_IDENT)
        }

        let mut productions = None;
        let mut generator = None;

        for attr in variant.attrs.iter() {
            let path = attr.path();
            if path.is_ident("to") {
                if productions.is_some() {
                    panic!("Exactly one #[to(...)] must be specified");
                }
                productions = Some(
                    attr.parse_args::<Productions>()
                        .expect("valid #[to(...)] format"),
                );
            } else if path.is_ident("param_gen") {
                if generator.is_some() {
                    panic!("Exactly one #[param_gen(...)] must be specified")
                }
                let gen_list = attr
                    .parse_args::<IdentList>()
                    .expect("#[param_gen(...)] arg is ident list separated by comma");
                if gen_list.elems.len() != 1 {
                    panic!("Exactly 1 generator must be specified with #[param_gen(...)]");
                }
                generator = Some(gen_list.elems[0].clone());
            }
        }

        let productions = productions.map(|productions| {
            // It's a non terminal.
            if variant.fields.len() > 0 {
                panic!("Non terminal can't have fields");
            }
            productions
                .elems
                .into_iter()
                .map(|production| {
                    if let either::Left(production) = production {
                        Some(
                            production
                                .into_iter()
                                .map(|it| {
                                    let index = index_by_variant.get(&it);
                                    if index.is_none() {
                                        panic!(
                                            "{} it's non enum variant, \
                                    every ident in the production right side \
                                    must be enum variant name",
                                            it
                                        );
                                    }
                                    *index.unwrap()
                                })
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        });

        variants.push(Variant {
            name: variant.ident.clone(),
            productions: productions,
            generator: generator,
            fields_count: variant.fields.len(),
        });
    }

    variants
}

/// Make get_productions(...) code.
///
/// Example:
/// fn get_productions<'a, 'b, 'c>(num: usize) -> &'b [Option<&'c [usize]>] {
///     match num {
///         0usize => &[Some(&[1usize, 2usize, 3usize]), None],
///         _ => {
///             panic!("get_productions(...) called on terminal!");
///         }
///     }
/// }
fn make_get_productions(variants: &Vec<Variant>) -> proc_macro2::TokenStream {
    let mut non_terminal_count = 0;

    let arms = variants
        .iter()
        .enumerate()
        .filter(|(_, v)| v.productions.is_some())
        .map(|(i, v)| {
            non_terminal_count += 1;
            let productions = v.productions.as_ref().unwrap();
            let productions = productions
                .into_iter()
                .map(|production| {
                    if let Some(production) = production {
                        // Non epsilon production.
                        let production_array = production.into_iter().map(|it| {
                            let num = *it;
                            quote! { #num }
                        });
                        quote! {
                            Some(&[#(#production_array),*])
                        }
                    } else {
                        // Epsilon production.
                        quote! {
                            None
                        }
                    }
                })
                .collect::<Vec<_>>();
            quote! {
                #i => &[#(#productions),*],
            }
        });

    quote! {
        fn get_productions<'a, 'b, 'c>(num: usize) -> &'b [Option<&'c[usize]>] {
            match num {
                #(#arms)*
                _ => panic!("get_productions(...) called on terminal!")
            }
        }
    }
}

/// Make is_terminal(...) code.
///
/// Example:
/// fn is_terminal(num: usize) -> bool {
///     match num {
///         1usize | 2usize | 3usize => true,
///         _ => false,
///     }
/// }
fn make_is_terminal(variants: &Vec<Variant>) -> proc_macro2::TokenStream {
    let terminals = variants
        .iter()
        .enumerate()
        .filter(|(_, v)| v.productions.is_none())
        .map(|(i, _)| quote! { #i })
        .collect::<Vec<_>>();
    quote! {
        fn is_terminal(num: usize) -> bool {
            match num {
             #(#terminals)|* => true ,
             _ => false,
            }
        }
    }
}

/// Make from_num(...) code.
///
/// Example:
/// fn from_num(num: usize) -> Self {
///     match num {
///         0usize => Self::S,
///         1usize => Self::A,
///         2usize => Self::B,
///         3usize => Self::Int(i32_generator()),
///         _ => {
///             panic!("can't transform from num! Consider using `#[param_gen(func_name)]`")
///         }
///     }
/// }
fn make_from_num(variants: &Vec<Variant>) -> proc_macro2::TokenStream {
    let arms = variants.iter().enumerate().map(|(i, v)| {
        let name = &v.name;
        if v.productions.is_some() {
            // It's a non terminal.
            assert!(v.fields_count == 0);
            quote! { #i =>  Self::#name, }
        } else {
            // It's a terminal.
            if v.fields_count == 0 {
                quote! {#i => Self::#name, }
            } else if let Some(generator) = &v.generator {
                quote! {#i => Self::#name(#generator()), }
            } else {
                quote! {#i => panic!("can't transform from num!\
                Consider using `#[param_gen(func_name)]`"),}
            }
        }
    });

    quote! {
        fn from_num(num: usize) -> Self {
            match num {
                #(#arms)*
                _ => panic!("wrong argument")
            }
        }
    }
}

/// Make to_num(...) code.
///
/// Example:
/// fn to_num(&self) -> usize {
///     match self {
///         Self::S => 0usize,
///         Self::A => 1usize,
///         Self::B => 2usize,
///         Self::Int(_) => 3usize,
///     }
/// }
fn make_to_num(variants: &Vec<Variant>) -> proc_macro2::TokenStream {
    let arms = variants.iter().enumerate().map(|(i, v)| {
        let name = &v.name;
        if v.fields_count == 0 {
            quote! { Self::#name => #i, }
        } else {
            let wild_cards = (0..v.fields_count).map(|_| quote! {_});
            quote! { Self::#name(  #(#wild_cards),*  ) => #i, }
        }
    });
    quote! {
        fn to_num(&self) -> usize {
            match self {
                #(#arms)*
            }
        }
    }
}

/// Check if grammar is iterable.
///
/// Grammar is iterable iff every terminal:
/// - hasn't fields OR
/// - has generator
/// If so, we can generate terminals from nums during iteration.
fn is_grammar_iterable(variants: &Vec<Variant>) -> bool {
    for v in variants {
        if v.productions.is_some() {
            continue;
        }
        if v.fields_count > 0 && v.generator.is_none() {
            return false;
        }
    }
    true
}

#[proc_macro_derive(GrammarSymbol, attributes(to, param_gen))]
pub fn derive_grammar_symbol_parse(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let data = if let syn::Data::Enum(body) = input.data {
        body
    } else {
        panic!("Only enum style grammatics are supported for now")
    };

    let variants = extract_variants(&data);
    if variants.is_empty() {
        panic!("Enum must be non-empty");
    }
    if !variants[0].productions.is_some() {
        panic!("First variant must be non terminal and has #[to(...)] attribute");
    }
    let get_productions_impl = make_get_productions(&variants);
    let is_terminal_impl = make_is_terminal(&variants);
    let from_num_impl = make_from_num(&variants);
    let to_num_impl = make_to_num(&variants);

    let self_name = input.ident;
    let iterable_impl = if is_grammar_iterable(&variants) {
        quote! { impl parus::grammar::grammar::IterableGrammarSymbol for #self_name {} }
    } else {
        quote! {}
    };
    quote! {
        impl parus::grammar::grammar::GrammarSymbol for #self_name {
            fn start_non_terminal() -> usize {
                0
            }
            #is_terminal_impl
            #from_num_impl
            #to_num_impl
            #get_productions_impl
        }
        #iterable_impl
    }
    .into()
}
