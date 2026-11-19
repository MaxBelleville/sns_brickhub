use crate::brick::Brick;
use sns_helper::prelude::{GpuBuffer, GpuBufferHandle, GpuBufferUsage, GpuDevice, GpuError};

#[derive(Default)]
pub struct Buffers {
    bricks: Option<GpuBuffer<Brick>>,
    materials: Option<GpuBuffer<u16>>,
    indices: Option<GpuBuffer<u32>>,
}

impl Buffers {
    pub fn ensure_capacity(
        &mut self,
        gpu: &dyn GpuDevice,
        brick_cap: usize,
        material_cap: usize,
        index_cap: usize,
    ) -> Result<(), GpuError> {
        ensure_buffer(&mut self.bricks, gpu, brick_cap)?;
        ensure_buffer(&mut self.materials, gpu, material_cap)?;
        ensure_buffer(&mut self.indices, gpu, index_cap)?;
        Ok(())
    }
    pub fn handles(&self) -> Option<GpuBufferHandles> {
        Some(GpuBufferHandles {
            bricks: self.bricks.as_ref()?.handle(),
            materials: self.materials.as_ref()?.handle(),
            indices: self.indices.as_ref()?.handle(),
        })
    }
    pub fn bricks(&self) -> &GpuBuffer<Brick> {
        self.bricks
            .as_ref()
            .expect("brick buffer capacity was not ensured")
    }
    pub fn materials(&self) -> &GpuBuffer<u16> {
        self.materials
            .as_ref()
            .expect("material buffer capacity was not ensured")
    }
    pub fn indices(&self) -> &GpuBuffer<u32> {
        self.indices
            .as_ref()
            .expect("index buffer capacity was not ensured")
    }
}

#[derive(Clone, Copy)]
pub struct GpuBufferHandles {
    pub bricks: GpuBufferHandle,
    pub materials: GpuBufferHandle,
    pub indices: GpuBufferHandle,
}

fn ensure_buffer<T>(
    buffer: &mut Option<GpuBuffer<T>>,
    gpu: &dyn GpuDevice,
    required: usize,
) -> Result<(), GpuError> {
    if required == 0 {
        return Ok(());
    }
    let current = buffer.as_ref().map_or(0, GpuBuffer::capacity);
    let capacity = if current >= required {
        return Ok(());
    } else {
        required
            .checked_next_power_of_two()
            .ok_or(GpuError::CapacityOverflow)?
    };
    let usage = GpuBufferUsage {
        storage: true,
        transfer_src: true,
        transfer_dst: true,
    };
    if let Some(existing) = buffer {
        let replacement = GpuBuffer::new(gpu, capacity, usage)?;
        let completion = match replacement.copy_from(gpu, existing, 0, 0, existing.capacity()) {
            Ok(completion) => completion,
            Err(error) => {
                let _ = replacement.destroy(gpu);
                return Err(error);
            }
        };
        if let Err(error) = completion.wait() {
            let _ = replacement.destroy(gpu);
            return Err(error);
        }
        let old = std::mem::replace(existing, replacement);
        old.destroy(gpu)?;
    } else {
        *buffer = Some(GpuBuffer::new(gpu, capacity, usage)?);
    }
    Ok(())
}
