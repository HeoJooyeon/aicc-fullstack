import React from 'react'
import {Box, Button, Container, Flex, Text, useColorMode, useColorModeValue} from "@chakra-ui/react"

const Navbar = () => {
  return (
    <Container maxW={"900px"}>
        <Box px={4} my={4} borderRadius={5} bg={useColorModeValue("gray.200", "gray.700")}>
        {/* px = padding */}
            <Flex h="16" alignItems={"center"} justifyContent={"space-between"}>
                {/* left */}
                <Flex alignItems={"center"} justifyContent={"space-between"} gap={3} display={{base:"none", sm:"flex"}}>
                    <img src='/react.png' alt='react' width={50} height={50} />
                    <Text fontSize={"40px"}>+</Text>
                    <img src='/python.png' alt='python' width={50} height={50} />
                    <Text fontSize={"40px"}>=</Text>
                    <img src='/react.png' alt='react' width={50} height={50} />
                </Flex>
            </Flex>
        </Box>
    </Container>    
  )
}

export default Navbar
