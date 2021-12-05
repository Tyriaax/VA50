from homography import *
from mouseInput import *
from pawns_recognition import *
from cards_recognition import *
from GameBoard import *
from drawing import *

class GameProcessor:
    capEveryFrame = False
    homographymatrixfound = False

    def __init__(self, img, window_name):
        self.window_name = window_name

        self.height = img.shape[0]
        self.width = img.shape[1]

        self.gameBoard = GameBoard()
        self.pawnsRecognitionHelper = PawnsRecognitionHelper(self.height, self.width, self.gameBoard)
        self.cardsRecognitionHelper = CardsRecognitionHelper(self.height, self.width, self.gameBoard)

    def ComputeFrame(self, img):
        if not self.homographymatrixfound:
            if len(list_board_coords) >= 4:
                self.homographymatrix, self.coordinates = get_homography_matrix(img, np.array(list_board_coords), self.width, self.height)

                self.cardsRecognitionHelper.GetScreenPortions(img, self.coordinates)
                self.pawnsRecognitionHelper.GetScreenPortion(img, self.coordinates)
                self.homographymatrixfound = True
                self.gameBoard.updateGameStatus()
        else:
            img = cv2.warpPerspective(img, self.homographymatrix, (img.shape[1], img.shape[0]))

        return img

    def DrawFrame(self, img):
        modifiedimg = img.copy()

        if len(list_board_coords) < 4:
            for coord in list_board_coords:
                cv2.circle(modifiedimg, coord, 10, (0, 255, 0), -1)
        else:
            if self.gameBoard.getGameStatus() == GameStates.GSWaitingHomography:
                modifiedimg = drawText(modifiedimg, "Veuillez selectionner les quatres coins des 9 cartes")
            if self.gameBoard.getGameStatus() == GameStates.GSWaitingFirstRecognition:
                modifiedimg = self.pawnsRecognitionHelper.DrawZonesRectangles(modifiedimg)
                modifiedimg = drawText(modifiedimg, "Veuillez appuyer sur espace pour lancer une nouvelle detection")
            if self.gameBoard.getGameStatus() == GameStates.GSGameStarted:
                modifiedimg = self.pawnsRecognitionHelper.DrawFrame(modifiedimg)
                modifiedimg = self.cardsRecognitionHelper.DrawFrame(modifiedimg)

        return modifiedimg

    def ComputeInputs(self, img):
        continuebool = True

        if len(list_board_coords) < 4:
            cv2.setMouseCallback(self.window_name, mousePoints)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            continuebool = False
        if (key == 32 or self.capEveryFrame) and (self.gameBoard.getGameStatus().value > GameStates.GSWaitingHomography.value):
            self.gameBoard.updateGameStatus()
            self.cardsRecognitionHelper.IsInLineOfSight(img, [], (0, 3), (1, 3))  # (1,0), (1,3))
            self.cardsRecognitionHelper.GetEmptySideCards(img)
            self.cardsRecognitionHelper.ComputeFrame(img)
            self.pawnsRecognitionHelper.ComputeFrame(img)
            self.gameBoard.printState()

        return continuebool