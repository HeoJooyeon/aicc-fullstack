import { Button, ButtonGroup, Container, Stack } from '@chakra-ui/react'
import Navbar from './components/Navbar'

function App() {  

  return (
    <Stack minH={"100vh"}>
      <Navbar />
      <Container maxW={"1200px"} my={4}> 
      {/* my = margin top / bottom */}

      </Container>
    </Stack>
  )
}

export default App
