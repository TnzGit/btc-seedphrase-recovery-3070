/*! utils.rs: Terminal banner used at startup. */

use colored::*;
use std::io::Write;
use std::thread::sleep;
use std::time::Duration;

pub fn print_header() {
    print!("\x1B[2J\x1B[1;1H");

    let art1 = r#"
   ███████╗███████╗███████╗██████╗
   ██╔════╝██╔════╝██╔════╝██╔══██╗
   ███████╗█████╗  █████╗  ██║  ██║
   ╚════██║██╔══╝  ██╔══╝  ██║  ██║
   ███████║███████╗███████╗██████╔╝
   ╚══════╝╚══════╝╚══════╝╚═════╝ "#;

    let art2 = r#"
   ██████╗ ███████╗ ██████╗ ██████╗ ██╗   ██╗███████╗██████╗ ██╗   ██╗
   ██╔══██╗██╔════╝██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝
   ██████╔╝█████╗  ██║     ██║   ██║██║   ██║█████╗  ██████╔╝ ╚████╔╝
   ██╔══██╗██╔══╝  ██║     ██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗  ╚██╔╝
   ██║  ██║███████╗╚██████╗╚██████╔╝ ╚████╔╝ ███████╗██║  ██║   ██║
   ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝   "#;

    for line in art1.lines() {
        println!("{}", line.cyan().bold());
        sleep(Duration::from_millis(30));
    }
    for line in art2.lines() {
        println!("{}", line.cyan().bold());
        sleep(Duration::from_millis(30));
    }

    println!("\n");
    print!("{}", "      Built by ".white());
    for c in "Zun".chars() {
        print!("{}", c.to_string().magenta().bold());
        std::io::stdout().flush().unwrap();
        sleep(Duration::from_millis(100));
    }
    println!("\n");

    print!("  Loading BTC Module... ");
    std::io::stdout().flush().unwrap();
    sleep(Duration::from_millis(80));
    println!("{}", "OK".green().bold());

    println!("\n{}", "  ==================================================================".yellow());
    println!("\n");
}
