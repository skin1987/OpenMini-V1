//! OpenMini Rust 客户端示例
//!
//! 演示如何使用 Rust gRPC 客户端与 OpenMini 服务进行交互。
//! 包括聊天、图像理解、流式响应等功能示例。

use tonic::transport::Channel;
use tonic::Request;
use std::io::Read;

pub mod openmini {
    tonic::include_proto!("openmini");
}

use openmini::{
    open_mini_client::OpenMiniClient,
    ChatRequest, Message, ImageRequest, HealthRequest,
};

/// 示例 1: 基本聊天
/// 
/// 演示如何发送文本消息并获取流式响应。
async fn example_chat(client: &mut OpenMiniClient<Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("示例 1: 基本聊天");
    println!("{}", "=".repeat(50));

    let request = Request::new(ChatRequest {
        session_id: String::new(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "你好，请用一句话介绍自己".to_string(),
            image_data: vec![],
            audio_data: vec![],
            video_data: vec![],
        }],
        stream: true,
        max_tokens: 256,
        temperature: 0.7,
    });

    println!("用户: 你好，请用一句话介绍自己");
    print!("助手: ");
    
    let response = client.chat(request).await?;
    let mut stream = response.into_inner();
    
    while let Some(response) = stream.message().await? {
        print!("{}", response.token);
        
        if response.finished {
            if let Some(usage) = response.usage {
                println!("\n[Token 使用: {}]", usage.total_tokens);
            }
        }
    }
    
    println!();
    Ok(())
}

/// 示例 2: 多轮对话
/// 
/// 演示如何在对话中保持上下文。
async fn example_multi_turn_chat(client: &mut OpenMiniClient<Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("示例 2: 多轮对话");
    println!("{}", "=".repeat(50));

    let request = Request::new(ChatRequest {
        session_id: "session-123".to_string(),
        messages: vec![
            Message {
                role: "user".to_string(),
                content: "我叫张三".to_string(),
                image_data: vec![],
                audio_data: vec![],
                video_data: vec![],
            },
            Message {
                role: "assistant".to_string(),
                content: "你好张三，很高兴认识你！".to_string(),
                image_data: vec![],
                audio_data: vec![],
                video_data: vec![],
            },
            Message {
                role: "user".to_string(),
                content: "你还记得我的名字吗？".to_string(),
                image_data: vec![],
                audio_data: vec![],
                video_data: vec![],
            },
        ],
        stream: true,
        max_tokens: 256,
        temperature: 0.7,
    });

    println!("用户: 我叫张三");
    println!("助手: 你好张三，很高兴认识你！");
    println!("用户: 你还记得我的名字吗？");
    print!("助手: ");
    
    let response = client.chat(request).await?;
    let mut stream = response.into_inner();
    
    while let Some(response) = stream.message().await? {
        print!("{}", response.token);
    }
    
    println!("\n");
    Ok(())
}

/// 示例 3: 图像理解
/// 
/// 演示如何发送图像并获取描述。
async fn example_image_understanding(client: &mut OpenMiniClient<Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("示例 3: 图像理解");
    println!("{}", "=".repeat(50));

    let image_path = "test_image.jpg";
    let image_data = match std::fs::read(image_path) {
        Ok(data) => data,
        Err(_) => {
            println!("提示: 未找到测试图片 {}", image_path);
            println!("请将图片放置在当前目录下");
            return Ok(());
        }
    };

    let request = Request::new(ImageRequest {
        session_id: String::new(),
        image_data,
        question: "请描述这张图片的内容".to_string(),
        stream: false,
    });

    println!("正在分析图片: {}", image_path);
    println!("问题: 请描述这张图片的内容");
    print!("回答: ");

    let response = client.image_understanding(request).await?;
    let result = response.into_inner();
    
    println!("{}", result.token);
    println!();
    Ok(())
}

/// 示例 4: 图像理解 (流式)
/// 
/// 演示如何流式获取图像理解结果。
async fn example_image_understanding_stream(client: &mut OpenMiniClient<Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("示例 4: 图像理解 (流式)");
    println!("{}", "=".repeat(50));

    let image_path = "test_image.jpg";
    let image_data = match std::fs::read(image_path) {
        Ok(data) => data,
        Err(_) => {
            println!("提示: 未找到测试图片 {}", image_path);
            return Ok(());
        }
    };

    let request = Request::new(ImageRequest {
        session_id: String::new(),
        image_data,
        question: "这张图片里有什么？".to_string(),
        stream: true,
    });

    println!("正在分析图片...");
    print!("回答: ");

    let response = client.image_understanding_stream(request).await?;
    let mut stream = response.into_inner();
    
    while let Some(response) = stream.message().await? {
        print!("{}", response.token);
    }
    
    println!("\n");
    Ok(())
}

/// 示例 5: 健康检查
/// 
/// 演示如何检查服务状态。
async fn example_health_check(client: &mut OpenMiniClient<Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("示例 5: 健康检查");
    println!("{}", "=".repeat(50));

    let request = Request::new(HealthRequest {});
    let response = client.health_check(request).await?;
    let result = response.into_inner();

    println!("服务状态: {}", if result.healthy { "正常" } else { "异常" });
    println!("消息: {}", result.message);
    println!();
    Ok(())
}

/// 示例 6: 批量请求
/// 
/// 演示如何发送多个独立请求。
async fn example_batch_requests(client: &mut OpenMiniClient<Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("示例 6: 批量请求");
    println!("{}", "=".repeat(50));

    let prompts = vec![
        "什么是人工智能？",
        "什么是机器学习？",
        "什么是深度学习？",
    ];

    for prompt in prompts {
        let request = Request::new(ChatRequest {
            session_id: String::new(),
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
                image_data: vec![],
                audio_data: vec![],
                video_data: vec![],
            }],
            stream: false,
            max_tokens: 128,
            temperature: 0.7,
        });

        println!("用户: {}", prompt);
        print!("助手: ");

        let response = client.chat(request).await?;
        let mut stream = response.into_inner();
        
        while let Some(response) = stream.message().await? {
            print!("{}", response.token);
        }
        
        println!("\n");
    }

    Ok(())
}

/// 示例 7: 错误处理
/// 
/// 演示如何处理常见的错误情况。
async fn example_error_handling(client: &mut OpenMiniClient<Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("示例 7: 错误处理");
    println!("{}", "=".repeat(50));

    let request = Request::new(ChatRequest {
        session_id: String::new(),
        messages: vec![Message {
            role: "user".to_string(),
            content: "测试消息".to_string(),
            image_data: vec![],
            audio_data: vec![],
            video_data: vec![],
        }],
        stream: true,
        max_tokens: 10,
        temperature: 0.7,
    });

    match client.chat(request).await {
        Ok(response) => {
            let mut stream = response.into_inner();
            while let Some(response) = stream.message().await? {
                if response.token.contains("[错误") {
                    println!("生成错误: {}", response.token);
                    break;
                }
                print!("{}", response.token);
            }
            println!();
        }
        Err(e) => {
            println!("gRPC 错误: {:?}", e);
        }
    }

    println!();
    Ok(())
}

/// 示例 8: 自定义参数
/// 
/// 演示如何使用不同的生成参数。
async fn example_custom_parameters(client: &mut OpenMiniClient<Channel>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(50));
    println!("示例 8: 自定义参数");
    println!("{}", "=".repeat(50));

    let configs = vec![
        ("确定性输出 (temperature=0.0)", 0.0),
        ("平衡输出 (temperature=0.7)", 0.7),
        ("创造性输出 (temperature=1.2)", 1.2),
    ];

    for (desc, temp) in configs {
        println!("\n配置: {}", desc);
        
        let request = Request::new(ChatRequest {
            session_id: String::new(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "写一个关于春天的句子".to_string(),
                image_data: vec![],
                audio_data: vec![],
                video_data: vec![],
            }],
            stream: true,
            max_tokens: 64,
            temperature: temp,
        });

        print!("助手: ");
        let response = client.chat(request).await?;
        let mut stream = response.into_inner();
        
        while let Some(response) = stream.message().await? {
            print!("{}", response.token);
        }
        println!();
    }

    println!();
    Ok(())
}

/// 运行所有示例
async fn run_all_examples() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(50));
    println!("OpenMini Rust 客户端示例");
    println!("{}\n", "=".repeat(50));

    let addr = "http://localhost:50051";
    println!("连接到: {}", addr);
    
    let mut client = match OpenMiniClient::connect(addr).await {
        Ok(c) => c,
        Err(e) => {
            println!("错误: 无法连接到服务");
            println!("请确保 OpenMini 服务已启动: {}", addr);
            println!("错误详情: {}", e);
            return Ok(());
        }
    };

    example_health_check(&mut client).await?;
    example_chat(&mut client).await?;
    example_multi_turn_chat(&mut client).await?;
    example_batch_requests(&mut client).await?;
    example_custom_parameters(&mut client).await?;
    example_error_handling(&mut client).await?;
    
    println!("\n提示: 图像理解示例需要测试图片 test_image.jpg");
    example_image_understanding(&mut client).await?;
    example_image_understanding_stream(&mut client).await?;

    println!("\n所有示例执行完成！");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_all_examples().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connection() {
        let addr = "http://localhost:50051";
        let result = OpenMiniClient::connect(addr).await;
        assert!(result.is_ok(), "应该能够连接到服务");
    }

    #[tokio::test]
    async fn test_health_check() {
        let addr = "http://localhost:50051";
        let mut client = OpenMiniClient::connect(addr)
            .await
            .expect("连接失败");

        let request = Request::new(HealthRequest {});
        let response = client.health_check(request).await;
        
        assert!(response.is_ok(), "健康检查应该成功");
        let result = response.unwrap().into_inner();
        assert!(result.healthy, "服务应该健康");
    }

    #[tokio::test]
    async fn test_chat() {
        let addr = "http://localhost:50051";
        let mut client = OpenMiniClient::connect(addr)
            .await
            .expect("连接失败");

        let request = Request::new(ChatRequest {
            session_id: String::new(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "你好".to_string(),
                image_data: vec![],
                audio_data: vec![],
                video_data: vec![],
            }],
            stream: true,
            max_tokens: 32,
            temperature: 0.7,
        });

        let response = client.chat(request).await;
        assert!(response.is_ok(), "聊天请求应该成功");
    }
}
