fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_files = &["../openmini-proto/proto/openmini.proto"];

    let mut config = prost_build::Config::new();
    config
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .out_dir("src/");

    config.compile_protos(proto_files, &["../openmini-proto/proto/"])?;

    println!("cargo:rerun-if-changed=../openmini-proto/proto/openmini.proto");

    Ok(())
}
