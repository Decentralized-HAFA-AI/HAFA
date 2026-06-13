// ============================================================================
// True Streaming Dataset (Memory-Efficient Iterator)
// ============================================================================
//
// Provides zero-copy, memory-efficient iteration over text data.
// Key features:
// - No data duplication (uses byte slices)
// - Supports random access via set_position() for shuffled training
// - Implements Iterator trait for idiomatic Rust usage
// - Prevents Memory Explosion (no collect() needed)
//
// ============================================================================

pub struct StreamingDataset<'a> {
    bytes: &'a [u8],
    context_size: usize,
    current_index: usize,
}

impl<'a> StreamingDataset<'a> {
    /// Creates a new streaming dataset from text
    pub fn new(text: &'a str, context_size: usize) -> Self {
        Self {
            bytes: text.as_bytes(),
            context_size,
            current_index: 0,
        }
    }
    
    /// Returns the total number of samples in the dataset
    pub fn len(&self) -> usize {
        if self.bytes.len() > self.context_size {
            self.bytes.len() - self.context_size
        } else {
            0
        }
    }
    
    /// Checks if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Resets the iterator to the beginning (for epoch restart)
    pub fn reset(&mut self) {
        self.current_index = 0;
    }
    
    /// Sets the current position in the dataset (for shuffled iteration)
    pub fn set_position(&mut self, index: usize) {
        if index >= self.len() {
            panic!(
                "Index {} out of bounds for StreamingDataset with length {}",
                index,
                self.len()
            );
        }
        self.current_index = index;
    }
    
    /// Returns the current position
    pub fn position(&self) -> usize {
        self.current_index
    }
}

impl<'a> Iterator for StreamingDataset<'a> {
    type Item = (&'a [u8], u8);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index + self.context_size < self.bytes.len() {
            let input = &self.bytes[self.current_index..self.current_index + self.context_size];
            let target = self.bytes[self.current_index + self.context_size];
            self.current_index += 1;
            Some((input, target))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_iteration() {
        let text = "Hello World";
        let dataset = StreamingDataset::new(text, 3);
        
        let samples: Vec<_> = dataset.collect();
        assert_eq!(samples.len(), 8);
        assert_eq!(samples[0], (b"Hel" as &[u8], b'l'));
    }

    #[test]
    fn test_set_position() {
        let text = "Hello World";
        let mut dataset = StreamingDataset::new(text, 3);
        
        dataset.set_position(5);
        assert_eq!(dataset.position(), 5);
        
        let (input, target) = dataset.next().unwrap();
        assert_eq!(input, b" Wo");
        assert_eq!(target, b'r');
    }

    #[test]
    fn test_reset() {
        let text = "Hello World";
        let mut dataset = StreamingDataset::new(text, 3);
        
        let _ = dataset.next();
        let _ = dataset.next();
        assert_eq!(dataset.position(), 2);
        
        dataset.reset();
        assert_eq!(dataset.position(), 0);
    }

    #[test]
    #[should_panic(expected = "Index 100 out of bounds")]
    fn test_out_of_bounds() {
        let text = "Hello";
        let mut dataset = StreamingDataset::new(text, 2);
        dataset.set_position(100);
    }
}