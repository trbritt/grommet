fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("proto/xt_prime.proto")?;
    println!("cargo:rerun-if-changed=proto/xt_prime.proto");
    Ok(())
}
